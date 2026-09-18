#!/usr/bin/env python3
"""B0 baseline: ask the model directly whether the image matches the question.

The probe experiments answer "can a linear readout of internal activations tell
Shuffled from Full/Blurred?".  This script answers the far simpler question that
has to be ruled out first: can the model just say so when asked?

Same rows as the probe (expanded_counterfactual_rows with the same shuffle seed
and canonical size), same positive class (Shuffled), same metrics, so the AUROC
lands in the same table as the activation probe.

Nothing is trained.  For the yes/no prompts the score is

    logP(no-tokens) - logP(yes-tokens)

read from the first answer position of a single forward pass, which makes it a
continuous score rather than an accuracy, directly comparable to a probe.

Writes results.jsonl incrementally; re-running resumes and skips finished rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from tqdm.auto import tqdm

import run_vlm_mi_experiments as R
from analyze_mismatch_activations import average_precision, binary_auc

CONDITIONS = ["full", "blurred", "shuffled"]

# {question} is substituted with the row's question text.
PROMPT_VARIANTS: dict[str, str] = {
    "V1": (
        "이 이미지는 아래 질문에 답하기 위한 이미지입니까?\n"
        "질문: {question}\n"
        '"예" 또는 "아니오"로만 답하세요.'
    ),
    "V2": (
        "아래 질문과 이 이미지가 서로 관련이 있습니까?\n"
        "질문: {question}\n"
        "예 또는 아니오로만 답하세요."
    ),
    "V3": (
        "이 이미지가 아래 질문과 얼마나 관련이 있는지 0에서 10 사이의 정수로만 답하세요.\n"
        "질문: {question}"
    ),
}

# V3 is graded by parsing the generated number; the rest read yes/no log-probs.
SCALE_VARIANTS = {"V3"}

YES_WORDS = ["예", "네", "Yes", "yes"]
NO_WORDS = ["아니오", "아니요", "아뇨", "아니", "No", "no"]


# ------------------------------------------------------------------- prompting


_ACTIVE_TEMPLATE = {"text": PROMPT_VARIANTS["V1"]}


def make_conversation_relevance(
    row: Mapping[str, Any],
    image_store: R.ImageStore,
    prompt_keys: Mapping[str, str],
    prompt_key_override: str | None = None,
):
    """Replace the QA prompt with the relevance question.

    Patched over R.make_conversation so runner.prepare() still handles image
    resizing, chat templating and position masks.
    """
    prompt_text = _ACTIVE_TEMPLATE["text"].format(
        question=str(row.get("question") or "")
    )
    user_content: list[dict[str, Any]] = []
    image = image_store.resolve(row)
    if image is not None:
        user_content.append({"type": "image", "image": image})
    user_content.append({"type": "text", "text": prompt_text})
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": R.PROMPT["system"]}]},
        {"role": "user", "content": user_content},
    ]
    return conversation, prompt_text


R.make_conversation = make_conversation_relevance


# --------------------------------------------------------------------- scoring


def first_token_ids(tokenizer: Any, words: Sequence[str]) -> list[int]:
    """First-token ids for each word, with and without a leading space."""
    ids: set[int] = set()
    for word in words:
        for surface in (word, " " + word):
            encoded = tokenizer.encode(surface, add_special_tokens=False)
            if encoded:
                ids.add(int(encoded[0]))
    return sorted(ids)


def last_prompt_position(prepared: R.PreparedInput) -> int:
    return int(
        torch.nonzero(prepared.masks["last"][0], as_tuple=False).flatten()[-1]
    )


def yes_no_score(
    runner: R.ExperimentRunner,
    row: Mapping[str, Any],
    yes_ids: Sequence[int],
    no_ids: Sequence[int],
) -> dict[str, Any]:
    prepared = runner.prepare(row)
    with torch.inference_mode():
        outputs = runner.model(**prepared.inputs, use_cache=False, return_dict=True)
    logits = outputs.logits[0, last_prompt_position(prepared)].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    log_yes = float(torch.logsumexp(log_probs[list(yes_ids)], dim=0).item())
    log_no = float(torch.logsumexp(log_probs[list(no_ids)], dim=0).item())
    top_id = int(torch.argmax(logits).item())
    return {
        # Higher score = "not related" = mismatch, so Shuffled is the positive class.
        "score": log_no - log_yes,
        "log_yes": log_yes,
        "log_no": log_no,
        "said_no": log_no > log_yes,
        "top_token": runner.tokenizer.decode([top_id]).strip(),
    }


def scale_score(runner: R.ExperimentRunner, row: Mapping[str, Any]) -> dict[str, Any]:
    prepared = runner.prepare(row)
    with torch.inference_mode():
        generated = runner.model.generate(
            **prepared.inputs, max_new_tokens=4, do_sample=False, use_cache=True
        )
    prompt_length = prepared.inputs["input_ids"].shape[1]
    text = runner.processor.batch_decode(
        generated[:, prompt_length:], skip_special_tokens=True
    )[0].strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    value = float(digits[:2]) if digits else float("nan")
    if value == value:  # not NaN
        value = min(value, 10.0)
    return {
        # Low relevance = mismatch, so negate to keep "higher = mismatch".
        "score": -value,
        "raw_text": text,
        "parsed_value": value,
        "said_no": (value <= 5.0) if value == value else False,
    }


# --------------------------------------------------------------------- metrics


def summarize(records: Sequence[Mapping[str, Any]], variant: str) -> dict[str, Any]:
    rows = [r for r in records if r["variant"] == variant]
    usable = [r for r in rows if r["score"] == r["score"]]  # drop NaN
    by_condition: dict[str, list[Mapping[str, Any]]] = {}
    for row in usable:
        by_condition.setdefault(row["condition"], []).append(row)

    def auc(positive: str, negatives: Sequence[str]) -> float | None:
        pool = by_condition.get(positive, []) + [
            r for c in negatives for r in by_condition.get(c, [])
        ]
        if not pool:
            return None
        scores = [float(r["score"]) for r in pool]
        labels = [r["condition"] == positive for r in pool]
        return binary_auc(scores, labels)

    pool_all = usable
    ap = None
    if pool_all:
        ap = average_precision(
            [float(r["score"]) for r in pool_all],
            [r["condition"] == "shuffled" for r in pool_all],
        )

    # "Correct" means: say no on Shuffled, say yes on Full/Blurred.  The 0-10
    # scale has no natural cutoff, so accuracy is only reported for yes/no.
    accuracy = None
    if usable and variant not in SCALE_VARIANTS:
        correct = sum(
            1 for r in usable if r["said_no"] == (r["condition"] == "shuffled")
        )
        accuracy = correct / len(usable)
    return {
        "variant": variant,
        "n": len(rows),
        "n_scored": len(usable),
        "auroc_shuffled_vs_full_blurred": auc("shuffled", ["full", "blurred"]),
        "auroc_shuffled_vs_full": auc("shuffled", ["full"]),
        "ap_shuffled_vs_full_blurred": ap,
        "binary_accuracy": accuracy,
        "said_no_rate": {
            condition: (
                sum(1 for r in items if r["said_no"]) / len(items) if items else None
            )
            for condition, items in sorted(by_condition.items())
        },
        "mean_score": {
            condition: (
                sum(float(r["score"]) for r in items) / len(items) if items else None
            )
            for condition, items in sorted(by_condition.items())
        },
    }


# ------------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-family", default=None)
    parser.add_argument("--input-data", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="b0_direct_query")
    parser.add_argument("--variants", default="V1,V2,V3")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--canonical-size", default="672,672")
    parser.add_argument("--shuffle-seed", type=int, default=17)
    parser.add_argument("--blur-radius", type=float, default=12.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-heads", default=None)
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in PROMPT_VARIANTS]
    if unknown:
        raise SystemExit(
            f"모르는 변형 {unknown}. 사용 가능: {sorted(PROMPT_VARIANTS)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / f"{args.output_name}.jsonl"
    summary_path = args.output_dir / f"{args.output_name}_summary.json"

    from transformers import AutoProcessor

    rows = R.read_input(args.input_data)
    dtype = R.resolve_dtype(args.dtype)
    family = R.infer_model_family(args.model_id, args.model_family)
    model = R.load_vlm_model(
        args.model_id,
        family,
        dtype=dtype,
        attn_implementation="sdpa",
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    layers, decoder_path = R.decoder_layers(model)
    num_heads = (
        int(args.num_heads)
        if args.num_heads
        else R.infer_num_attention_heads(model, layers)
    )
    runner = R.ExperimentRunner(
        model=model,
        processor=processor,
        rows=rows,
        image_store=R.ImageStore({"original": args.image_root}),
        output_dir=args.output_dir,
        prompt_keys={"MC": "MC_1", "SA": "SA_1", "LA": "LA_1"},
        num_heads=num_heads,
        dtype=dtype,
        seed=args.shuffle_seed,
    )

    yes_ids = first_token_ids(runner.tokenizer, YES_WORDS)
    no_ids = first_token_ids(runner.tokenizer, NO_WORDS)
    overlap = sorted(set(yes_ids) & set(no_ids))
    if overlap:
        raise SystemExit(
            "예/아니오 첫 토큰이 겹칩니다: "
            f"{[runner.tokenizer.decode([i]) for i in overlap]}. "
            "YES_WORDS/NO_WORDS를 조정하세요."
        )
    print(f"yes tokens: {[runner.tokenizer.decode([i]) for i in yes_ids]}", flush=True)
    print(f"no  tokens: {[runner.tokenizer.decode([i]) for i in no_ids]}", flush=True)

    canonical = [int(v) for v in args.canonical_size.split(",")]
    expanded = runner.expanded_counterfactual_rows(
        CONDITIONS,
        limit=args.limit,
        blur_radius=args.blur_radius,
        shuffle_seed=args.shuffle_seed,
        canonical_image_size=canonical,
    )
    print(
        f"문항 {len(expanded) // len(CONDITIONS)} × 조건 {len(CONDITIONS)} "
        f"× 변형 {len(variants)} = {len(expanded) * len(variants)}회 forward",
        flush=True,
    )

    done: dict[str, dict[str, Any]] = {}
    if jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    done[record["record_id"]] = record
    if done:
        print(f"기존 결과 {len(done)}건을 이어받습니다: {jsonl_path}", flush=True)

    records: list[dict[str, Any]] = list(done.values())
    pending = [
        (variant, row)
        for variant in variants
        for row in expanded
        if f"{row.get('question_id')}::{R.norm_condition(row.get('condition'))}::{variant}"
        not in done
    ]

    with jsonl_path.open("a", encoding="utf-8") as stream:
        for variant, row in tqdm(pending, desc=args.output_name):
            condition = R.norm_condition(row.get("condition"))
            record_id = f"{row.get('question_id')}::{condition}::{variant}"
            _ACTIVE_TEMPLATE["text"] = PROMPT_VARIANTS[variant]
            if variant in SCALE_VARIANTS:
                scored = scale_score(runner, row)
            else:
                scored = yes_no_score(runner, row, yes_ids, no_ids)
            record = {
                "record_id": record_id,
                "question_id": str(row.get("question_id")),
                "condition": condition,
                "variant": variant,
                "shuffled_from_question_id": row.get("shuffled_from_question_id", ""),
                **scored,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            records.append(record)

    summaries = [summarize(records, variant) for variant in variants]
    summary_path.write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "input_data": str(args.input_data),
                "shuffle_seed": args.shuffle_seed,
                "canonical_size": canonical,
                "positive_class": "shuffled",
                "trained": False,
                "variants": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== B0 직접 질의 baseline ===")
    for summary in summaries:
        auroc = summary["auroc_shuffled_vs_full_blurred"]
        auroc_full = summary["auroc_shuffled_vs_full"]
        accuracy = summary["binary_accuracy"]
        print(f"\n[{summary['variant']}]  n={summary['n_scored']}")
        if auroc is None:
            print("  AUROC (shuffled vs full+blurred) = 계산 불가")
        else:
            print(f"  AUROC (shuffled vs full+blurred) = {auroc:.4f}")
        if auroc_full is not None:
            print(f"  AUROC (shuffled vs full)         = {auroc_full:.4f}")
        if accuracy is not None:
            print(f"  이진 정확도                      = {accuracy:.4f}")
        print(f"  조건별 '아니오' 비율: {summary['said_no_rate']}")
    print(f"\n요약: {summary_path}")
    print(f"원자료: {jsonl_path}")


if __name__ == "__main__":
    main()
