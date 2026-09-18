#!/usr/bin/env python3
"""Test-split inference and submission file, with optional retrieval gating.

Free generation for MC/SA/LA, then the official submission schema:

    [{"metadata": {question_id, task_type, corpus_name, split, question_form},
      "model_input": {image_name, question, options},
      "model_output": {"answer": ...}}, ...]

Retrieval gating (--retrieval-probe): the retrieved rank-1 image is attached and
the prefill activations are read in the same forward pass that produces the
answer.  If the frozen probe judges the pair mismatched, the answer is
regenerated with the original image alone.  That costs one extra generation only
for rejected items.

Without --retrieval-probe the script is a plain baseline runner, so the same
file produces both the control submission and the gated one.

Writes results.jsonl incrementally; re-running resumes and skips finished rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from tqdm.auto import tqdm

import run_vlm_mi_experiments as R
from retrieval_eval import (  # noqa: F401  (import patches make_conversation)
    IMAGE_NOTICE,
    _load_image,
    head_split,
    load_prompt_overrides,
    make_conversation_with_extras,
    read_csv,
    retrieval_index,
)

MAX_NEW_TOKENS = {"MC": 8, "SA": 64, "LA": 384}


def image_captions(n_retrieved: int) -> list[str]:
    """[문제 이미지] 다음에 [참고 이미지1..N] 표지를 붙인다."""
    if n_retrieved <= 1:
        return ["[문제 이미지]\n", "[참고 이미지 - 관련 없을 수 있음]\n"]
    return ["[문제 이미지]\n"] + [
        f"[참고 이미지{index} - 관련 없을 수 있음]\n"
        for index in range(1, n_retrieved + 1)
    ]


# ------------------------------------------------------------------ submission


def parse_options(row: Mapping[str, Any]) -> list[str]:
    try:
        options = json.loads(row.get("options_json") or "[]")
    except json.JSONDecodeError as error:
        raise ValueError(
            f"잘못된 options_json: question_id={row.get('question_id')}"
        ) from error
    return [str(option) for option in options]


def to_submission_sample(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "metadata": {
            "question_id": str(row.get("question_id", "")),
            "task_type": row.get("task_type", ""),
            "corpus_name": row.get("corpus_name", ""),
            "split": row.get("split", "test") or "test",
            "question_form": row.get("question_form", ""),
        },
        "model_input": {
            "image_name": row.get("input_image_name", ""),
            "question": row.get("question", ""),
            "options": parse_options(row),
        },
        "model_output": {
            "answer": str(row.get("prediction", row.get("output", ""))).strip(),
        },
    }


def write_submission_json(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    samples = [to_submission_sample(row) for row in rows]
    ids = [s["metadata"]["question_id"] for s in samples]
    if any(not i for i in ids):
        raise ValueError("Submission contains an empty question_id")
    if len(ids) != len(set(ids)):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"Submission contains duplicate question_id: {duplicates[:5]}")
    empty = [s["metadata"]["question_id"] for s in samples if not s["model_output"]["answer"]]
    if empty:
        raise ValueError(f"Submission contains an empty answer: {empty[:5]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------------------ probe


class RetrievalGate:
    def __init__(self, artifact: Mapping[str, Any], policy: str | None, invert: bool = False):
        # invert=True 면 probe 가 거부한 후보만 붙인다. probe 의 정합성 판정이
        # downstream 유용성과 음의 상관을 갖는지 검증하기 위한 대조 정책이다.
        self.invert = bool(invert)
        self.model = artifact["model"]
        self.heads = [(int(l), int(h)) for l, h in artifact["heads"]]
        self.layers = [int(v) for v in artifact["layers"]]
        self.num_heads = int(artifact["num_heads"])
        policy = policy or artifact.get("default_threshold_policy", "youden")
        thresholds = artifact["thresholds"]
        if policy not in thresholds:
            raise SystemExit(
                f"threshold policy {policy!r} 없음. 사용 가능: {sorted(thresholds)}"
            )
        self.threshold = float(thresholds[policy])
        self.policy = policy
        self.layer_pos = {layer: i for i, layer in enumerate(self.layers)}

    def score(self, stacked: torch.Tensor) -> float:
        vectors = [
            stacked[self.layer_pos[l], h].float().flatten() for l, h in self.heads
        ]
        values = torch.cat(vectors).tolist()
        means = [float(v) for v in self.model["means"]]
        scales = [float(v) for v in self.model["scales"]]
        weights = [float(v) for v in self.model["weights"]]
        logit = float(self.model["bias"]) + sum(
            w * ((v - m) / s) for v, m, s, w in zip(values, means, scales, weights)
        )
        return 1.0 / (1.0 + math.exp(-logit)) if logit >= 0 else (
            math.exp(logit) / (1.0 + math.exp(logit))
        )

    def attach(self, score: float) -> bool:
        aligned = score < self.threshold
        return (not aligned) if self.invert else aligned


# ---------------------------------------------------------------------- runner


class PrefillHeadCapture(R.MultiLayerHeadCapture):
    """Keep the prefill capture across generate()'s decode steps.

    The base hook re-fires on every decode step and, because
    position_mask_for_hidden returns None outside prefill, overwrites each layer
    with an empty tensor.  Reading activations from a generate() call therefore
    yields nothing unless the first non-empty capture is pinned.
    """

    def __enter__(self):
        result = super().__enter__()
        self._frozen: dict[int, torch.Tensor] = {}
        original = self.values

        class _PinnedDict(dict):
            def __setitem__(inner, key, value):  # noqa: N805
                if key in inner and inner[key].numel() > 0 and value.numel() == 0:
                    return
                dict.__setitem__(inner, key, value)

        self.values = _PinnedDict(original)
        return result


def generate_answer(
    runner: R.ExperimentRunner,
    row: Mapping[str, Any],
    max_new_tokens: int,
    capture_layers: Sequence[int] | None = None,
) -> tuple[str, dict[int, torch.Tensor] | None]:
    prepared = runner.prepare(row)
    hook = (
        PrefillHeadCapture(runner.model, prepared, list(capture_layers), "last")
        if capture_layers
        else None
    )
    context = hook if hook is not None else _NullContext()
    with context:
        with torch.inference_mode():
            generated = runner.model.generate(
                **prepared.inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
    prompt_length = prepared.inputs["input_ids"].shape[1]
    text = runner.processor.batch_decode(
        generated[:, prompt_length:], skip_special_tokens=True
    )[0].strip()
    values = hook.values if hook is not None else None
    return text, values


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--model-family", default="qwen3_vl")
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--adapter-path", default=None, help="LoRA adapter, optional")
    parser.add_argument("--input-data", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="submission")
    parser.add_argument("--mc-prompt-key", default="MC_1")
    parser.add_argument("--sa-prompt-key", default="SA_1")
    parser.add_argument("--la-prompt-key", default="LA_1")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help="prompts.py 또는 JSON. 생략하면 스크립트 폴더의 prompts.py 자동 사용.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--canonical-size", default="672,672")
    # retrieval gating
    parser.add_argument("--retrieval-manifest", type=Path, default=None)
    parser.add_argument("--retrieval-image-root", type=Path, default=None)
    parser.add_argument("--retrieval-probe", type=Path, default=None)
    parser.add_argument("--threshold-policy", default=None)
    parser.add_argument(
        "--invert-gate",
        action="store_true",
        help="probe 가 거부한 후보만 붙인다 (음의 상관 검증용 대조 정책).",
    )
    parser.add_argument(
        "--attach-always",
        action="store_true",
        help="Attach the retrieved image without gating (naive control run)",
    )
    parser.add_argument(
        "--max-attach",
        type=int,
        default=1,
        help="후보 상위 N개까지 개별 판정해 통과한 것만 붙인다 (기본 1).",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=None,
        help="검색 유사도가 이 값 미만인 후보는 판정 없이 제외한다.",
    )
    parser.add_argument(
        "--image-notice",
        action="store_true",
        help="이미지 역할 안내문과 [문제/참고 이미지] 표지를 프롬프트에 넣는다.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / f"{args.output_name}.jsonl"
    csv_path = args.output_dir / f"{args.output_name}.csv"
    submission_path = args.output_dir / f"{args.output_name}_submission.json"

    # 러너의 PROMPT는 모듈 전역 dict라 런타임에 덮어쓸 수 있다. 대회에서 쓰던
    # 출력 형식 규칙(복수정답 "/" 구분 등)이 여기에 들어간다.
    load_prompt_overrides(args.prompts)

    prompt_keys = {
        "MC": args.mc_prompt_key,
        "SA": args.sa_prompt_key,
        "LA": args.la_prompt_key,
    }
    for form, key in prompt_keys.items():
        R.validate_prompt_key(form, key)
    print(f"prompt keys: {prompt_keys}", flush=True)

    canonical = (
        [int(v) for v in args.canonical_size.split(",")] if args.canonical_size else None
    )

    # ---- model ------------------------------------------------------------
    from transformers import AutoProcessor

    rows = R.read_input(args.input_data)
    if args.limit:
        rows = rows[: args.limit]
    unsupported = sorted({r.get("question_form", "") for r in rows} - {"MC", "SA", "LA"})
    if unsupported:
        raise SystemExit(f"지원하지 않는 question_form: {unsupported}")

    dtype = R.resolve_dtype(args.dtype)
    family = R.infer_model_family(args.model_id, args.model_family)
    model = R.load_vlm_model(
        args.model_id,
        family,
        dtype=dtype,
        attn_implementation="sdpa",
        device_map="auto",
        trust_remote_code=True,
    )
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=False)
        print(f"LoRA adapter loaded: {args.adapter_path}", flush=True)
    model = model.eval()
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    layers, decoder_path = R.decoder_layers(model)
    num_heads = int(args.num_heads) if args.num_heads else R.infer_num_attention_heads(model, layers)
    runner = R.ExperimentRunner(
        model=model,
        processor=processor,
        rows=rows,
        image_store=R.ImageStore({"original": args.image_root}),
        output_dir=args.output_dir,
        prompt_keys=prompt_keys,
        num_heads=num_heads,
        dtype=dtype,
    )
    print(f"rows={len(rows)}  decoder={decoder_path}  heads={num_heads}", flush=True)

    # ---- retrieval --------------------------------------------------------
    index: dict[str, list[dict[str, Any]]] = {}
    gate: RetrievalGate | None = None
    if args.retrieval_manifest and args.retrieval_image_root:
        index = retrieval_index(
            read_csv(args.retrieval_manifest), args.split, args.retrieval_image_root
        )
        print(f"검색 이미지 확보 문항: {len(index)}", flush=True)
    if args.retrieval_probe:
        artifact = json.loads(args.retrieval_probe.read_text(encoding="utf-8"))
        gate = RetrievalGate(artifact, args.threshold_policy, args.invert_gate)
        print(
            f"gate: policy={gate.policy}{' INVERTED' if gate.invert else ''} "
            f"threshold={gate.threshold:.4f} "
            f"heads={len(gate.heads)} cv_auroc={artifact.get('cv_auroc')}",
            flush=True,
        )

    # ---- resume -----------------------------------------------------------
    done: dict[str, dict[str, Any]] = {}
    if jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    done[str(record["question_id"])] = record
    print(f"already done: {len(done)}", flush=True)

    stats = {"attached": 0, "rejected": 0, "no_candidate": 0, "images_attached": 0}
    pending = [r for r in rows if str(r["question_id"]) not in done]
    with jsonl_path.open("a", encoding="utf-8") as stream:
        for row in tqdm(pending, desc=args.output_name):
            qid = str(row["question_id"])
            form = R.norm_form(row.get("question_form"))
            budget = MAX_NEW_TOKENS[form]
            candidates = index.get(qid, [])[: max(0, args.max_attach)]

            base = dict(row)
            base["_target_image_size"] = canonical
            base["_extra_images"] = []
            decision = "orig_only"
            selected: list[dict[str, Any]] = []
            per_candidate: list[dict[str, Any]] = []

            for candidate in candidates:
                verdict: dict[str, Any] = {
                    "rank": candidate.get("rank"),
                    "similarity": candidate.get("similarity"),
                    "probe_score": "",
                    "attached": False,
                    "reason": "",
                }
                similarity = candidate.get("similarity")
                if (
                    args.min_similarity is not None
                    and similarity is not None
                    and float(similarity) < args.min_similarity
                ):
                    verdict["reason"] = "below_min_similarity"
                elif args.attach_always or gate is None:
                    verdict["attached"] = True
                    verdict["reason"] = "attach_always" if args.attach_always else "no_gate"
                else:
                    # 후보마다 원본과 짝지어 한 번만 prefill 해서 probe 점수를 읽는다.
                    probe_row = dict(base)
                    probe_row["_extra_images"] = [candidate["path"]]
                    _, values = generate_answer(runner, probe_row, 1, gate.layers)
                    stacked = head_split(values or {}, gate.layers, num_heads)
                    if stacked is None:
                        verdict["reason"] = "no_activation"
                    else:
                        score = gate.score(stacked)
                        verdict["probe_score"] = score
                        verdict["attached"] = gate.attach(score)
                        verdict["reason"] = "probe_pass" if verdict["attached"] else "probe_reject"
                if verdict["attached"]:
                    selected.append(candidate)
                per_candidate.append(verdict)

            if selected:
                final = dict(base)
                final["_extra_images"] = [c["path"] for c in selected]
                if args.image_notice:
                    final["_prompt_prefix"] = IMAGE_NOTICE
                    final["_image_captions"] = image_captions(len(selected))
                text, _ = generate_answer(runner, final, budget)
                decision = "attached_always" if args.attach_always else "attached"
                stats["attached"] += 1
                stats["images_attached"] += len(selected)
            else:
                if not candidates:
                    stats["no_candidate"] += 1
                elif per_candidate:
                    decision = "rejected"
                    stats["rejected"] += 1
                text, _ = generate_answer(runner, base, budget)

            record = dict(row)
            for key in ("_extra_images", "_target_image_size", "_prompt_prefix", "_image_captions"):
                record.pop(key, None)
            record.update(
                {
                    "prediction": text,
                    "output": text,
                    "prompt_key": prompt_keys[form],
                    "retrieval_decision": decision,
                    "retrieval_probe_score": (
                        per_candidate[0]["probe_score"] if per_candidate else ""
                    ),
                    "n_candidates": len(candidates),
                    "n_attached": len(selected),
                    "attached_ranks": [c.get("rank") for c in selected],
                    "attached_files": [Path(c["path"]).name for c in selected],
                    "candidate_verdicts": per_candidate,
                    "split": row.get("split") or args.split,
                }
            )
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            done[qid] = record

    order = {str(r["question_id"]): i for i, r in enumerate(rows)}
    results = sorted(done.values(), key=lambda r: order.get(str(r["question_id"]), 10**9))
    results = [r for r in results if str(r["question_id"]) in order]

    write_csv_rows(csv_path, results)
    write_submission_json(submission_path, results)
    print(f"\nsaved jsonl : {jsonl_path}")
    print(f"saved csv   : {csv_path}")
    print(f"saved submit: {submission_path}  ({len(results)}건)")
    if index:
        print(
            f"retrieval: attached {stats['attached']}, rejected {stats['rejected']}, "
            f"no_candidate {stats['no_candidate']}, "
            f"images_attached {stats['images_attached']}"
        )


if __name__ == "__main__":
    main()