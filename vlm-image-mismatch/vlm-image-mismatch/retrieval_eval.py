#!/usr/bin/env python3
"""Retrieval-augmented evaluation: does attaching a retrieved image help?

Conditions per question (MC only):

  text_only      question only
  orig           original image                     <- current baseline
  orig_top1      original + rank-1 retrieved image
  orig_shuffled  original + a retrieved image from ANOTHER question  <- control
  ret_only       rank-1 retrieved image, no original

orig vs orig_top1 is the go/no-go: their own numbers say a wrong image costs
-27.19pp against text-only while a right one gains +11.65pp, so naive attachment
loses unless retrieval is right ~70% of the time.  orig_shuffled bounds the
damage and supplies the mismatch class for a two-image probe.

Head activations at the last prompt token are saved for every condition so the
probe can be fitted afterwards without another GPU pass.

The runner is not modified.  make_conversation is wrapped so a row carrying
`_extra_images` (list of paths) appends them after the primary image.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image, ImageOps
from tqdm import tqdm

import run_vlm_mi_experiments as R


# --------------------------------------------------------------- multi-image

_ORIGINAL_MAKE_CONVERSATION = R.make_conversation


def _load_image(path: Path, canonical: tuple[int, int] | None) -> Image.Image:
    with Image.open(path) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")
    if canonical:
        # Letterbox so every condition keeps the same visual token count.
        target_w, target_h = canonical
        image.thumbnail((target_w, target_h), Image.BICUBIC)
        canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        canvas.paste(
            image, ((target_w - image.width) // 2, (target_h - image.height) // 2)
        )
        image = canvas
    return image


def make_conversation_with_extras(row, image_store, prompt_keys, prompt_key_override=None):
    conversation, prompt_text = _ORIGINAL_MAKE_CONVERSATION(
        row, image_store, prompt_keys, prompt_key_override
    )
    prefix = str(row.get("_prompt_prefix") or "")
    if prefix:
        user = conversation[-1]["content"]
        for item in user:
            if item.get("type") == "text":
                item["text"] = prefix + item["text"]
                prompt_text = prefix + prompt_text
                break
    extras = row.get("_extra_images") or []
    captions = row.get("_image_captions") or []
    if not extras and not captions:
        return conversation, prompt_text
    canonical = row.get("_target_image_size")
    canonical = tuple(canonical) if isinstance(canonical, (list, tuple)) else None
    user = conversation[-1]["content"]
    insert_at = sum(1 for item in user if item.get("type") == "image")
    for path in extras:
        user.insert(insert_at, {"type": "image", "image": _load_image(Path(path), canonical)})
        insert_at += 1
    if captions:
        # Interleave a marker before each image so the role is positional rather
        # than inferred from a preamble that sits after both images.
        rebuilt: list[dict[str, Any]] = []
        seen = 0
        for item in user:
            if item.get("type") == "image":
                if seen < len(captions) and captions[seen]:
                    rebuilt.append({"type": "text", "text": captions[seen]})
                seen += 1
            rebuilt.append(item)
        conversation[-1]["content"] = rebuilt
    return conversation, prompt_text


R.make_conversation = make_conversation_with_extras


def load_prompt_overrides(explicit: Path | None = None) -> dict[str, str]:
    """러너의 DEFAULT_PROMPTS를 대회용 프롬프트로 덮어쓴다.

    explicit 이 .py 면 그 모듈의 PROMPT, .json 이면 그 내용을 쓴다.
    지정이 없으면 스크립트 폴더의 prompts.py 를 자동으로 찾는다.
    """
    candidate = explicit
    if candidate is None:
        auto = Path(__file__).resolve().parent / "prompts.py"
        candidate = auto if auto.is_file() else None
    if candidate is None:
        return {}

    candidate = Path(candidate)
    if candidate.suffix == ".json":
        overrides = json.loads(candidate.read_text(encoding="utf-8"))
    else:
        import importlib.util

        spec = importlib.util.spec_from_file_location("_prompts", candidate)
        if spec is None or spec.loader is None:
            raise SystemExit(f"prompts 모듈을 불러올 수 없습니다: {candidate}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        overrides = getattr(module, "PROMPT", None)
        if not isinstance(overrides, dict):
            raise SystemExit(f"{candidate} 에 PROMPT dict 가 없습니다")

    overrides = {str(k): str(v) for k, v in overrides.items()}
    R.PROMPT.update(overrides)
    print(f"prompt overrides: {sorted(overrides)} <- {candidate}", flush=True)
    return overrides


# ------------------------------------------------------------------ manifest


def read_csv(path: Path) -> list[dict[str, str]]:
    with io.open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def retrieval_index(
    manifest_rows: Sequence[Mapping[str, str]], split: str, image_root: Path
) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for row in manifest_rows:
        if row.get("split") != split:
            continue
        name = str(row.get("image_name") or "")
        if not name or not (image_root / name).is_file():
            continue
        index.setdefault(str(row["question_id"]), []).append(
            {
                "rank": int(row["rank"]),
                "path": image_root / name,
                "title": row.get("title", ""),
                "description": row.get("description", ""),
                "similarity": float(row["similarity"]) if row.get("similarity") else None,
            }
        )
    for values in index.values():
        values.sort(key=lambda c: c["rank"])
    return index


def text_index(
    manifest_rows: Sequence[Mapping[str, str]], split: str
) -> dict[str, list[dict[str, Any]]]:
    """Every candidate, image or not — e뮤지엄 rows carry 표제어/설명 with no image."""
    index: dict[str, list[dict[str, Any]]] = {}
    for row in manifest_rows:
        if row.get("split") != split:
            continue
        index.setdefault(str(row["question_id"]), []).append(
            {
                "rank": int(row["rank"]),
                "title": row.get("title", ""),
                "description": row.get("description", ""),
                "similarity": float(row["similarity"]) if row.get("similarity") else None,
            }
        )
    for values in index.values():
        values.sort(key=lambda c: c["rank"])
    return index


IMAGE_NOTICE = (
    "[이미지 안내] 첫 번째 이미지는 질문에 해당하는 이미지입니다. "
    "두 번째 이미지는 참고용으로 검색된 자료이며 질문과 관련이 없을 수 있습니다. "
    "관련이 없다고 판단되면 무시하고 첫 번째 이미지만 사용하세요.\n\n"
)

# 안내 문구 변형. v1은 위 IMAGE_NOTICE와 동일(이미 측정됨).
# prefix  = 프롬프트 앞에 붙는 안내문
# markers = 각 이미지 바로 앞에 삽입되는 표지 [문제 이미지] / [참고 이미지]
NOTICE_VARIANTS: dict[str, dict[str, Any]] = {
    "v1": {"prefix": IMAGE_NOTICE, "markers": None},
    # 정답 근거를 첫 이미지로 못박고, 두 번째의 역할을 배경으로 한정
    "v2": {
        "prefix": (
            "[이미지 안내] 정답은 반드시 첫 번째 이미지를 근거로 판단하세요. "
            "두 번째 이미지는 배경 참고용일 뿐이며 정답의 근거가 될 수 없습니다.\n\n"
        ),
        "markers": None,
    },
    # 기본값을 '무시'로 두고, 명백히 도움이 될 때만 사용하도록
    "v3": {
        "prefix": (
            "[이미지 안내] 두 번째 이미지는 자동 검색된 자료로 대부분 질문과 무관합니다. "
            "기본적으로 무시하고, 첫 번째 이미지만으로 답하세요. "
            "두 번째 이미지가 명백히 같은 대상을 담고 있을 때만 보조로 참고하세요.\n\n"
        ),
        "markers": None,
    },
    # 안내문 없이, 각 이미지 앞에 표지만 삽입 (역할을 위치로 고정)
    "v4": {
        "prefix": "",
        "markers": ["[문제 이미지]\n", "[참고 이미지 - 관련 없을 수 있음]\n"],
    },
    # 표지 + 강한 안내문 병행
    "v5": {
        "prefix": (
            "[이미지 안내] 아래 이미지 중 [문제 이미지]로 표시된 것만이 질문의 대상입니다. "
            "[참고 이미지]는 검색으로 딸려온 자료이며 무관할 수 있으니, "
            "확신이 없으면 무시하세요.\n\n"
        ),
        "markers": ["[문제 이미지]\n", "[참고 이미지 - 관련 없을 수 있음]\n"],
    },
}


def evidence_text(candidates, limit: int = 3) -> str:
    """표제어/설명을 참고 자료 블록으로 만든다. 이미지가 없는 후보도 포함한다."""
    lines = []
    for candidate in candidates[:limit]:
        title = str(candidate.get("title") or "").strip()
        description = str(candidate.get("description") or "").strip()
        if not title and not description:
            continue
        lines.append(f"- {title}: {description}" if description else f"- {title}")
    if not lines:
        return ""
    return (
        "[참고 자료] 아래는 검색된 자료이며 질문과 관련이 없을 수 있습니다. "
        "도움이 될 때만 사용하고, 관련이 없으면 무시하세요.\n"
        + "\n".join(lines)
        + "\n\n"
    )


CONDITIONS = (
    "text_only",
    "orig",
    "orig_top1",
    "orig_shuffled",
    "ret_only",
    # 프롬프트로 역할을 명시한 변형과 텍스트 근거 주입
    "orig_top1_labeled",
    "orig_plus_text",
    "orig_plus_both",
) + tuple(f"orig_top1_{name}" for name in NOTICE_VARIANTS)


def build_runner(args: argparse.Namespace) -> R.ExperimentRunner:
    """Mirror of the construction block in run_vlm_mi_experiments.main()."""
    from transformers import AutoProcessor

    rows = R.read_input(args.input_data)
    dtype = R.resolve_dtype(args.dtype)
    family = R.infer_model_family(args.model_id, args.model_family)
    image_store = R.ImageStore({"original": args.image_root})
    print(f"rows={len(rows)}  model={args.model_id} (family={family})", flush=True)
    print(f"image_counts={image_store.counts()}", flush=True)

    model = R.load_vlm_model(
        args.model_id,
        family,
        dtype=dtype,
        attn_implementation="sdpa",
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    layers, decoder_path = R.decoder_layers(model)
    num_heads = (
        int(args.num_heads)
        if args.num_heads
        else R.infer_num_attention_heads(model, layers)
    )
    print(
        f"decoder={decoder_path}, layers={len(layers)}, query_heads={num_heads}",
        flush=True,
    )
    # 제출(submit_test.py)과 반드시 같은 프롬프트를 써야 검증 결과가 전이된다.
    prompt_keys = {
        "MC": args.mc_prompt_key,
        "SA": args.sa_prompt_key,
        "LA": args.la_prompt_key,
    }
    for form, key in prompt_keys.items():
        R.validate_prompt_key(form, key)
    print(f"prompt keys: {prompt_keys}", flush=True)

    runner = R.ExperimentRunner(
        model=model,
        processor=processor,
        rows=rows,
        image_store=image_store,
        output_dir=args.output_dir,
        prompt_keys=prompt_keys,
        num_heads=num_heads,
        dtype=dtype,
        seed=args.shuffle_seed,
    )
    runner.n_layers = len(layers)
    return runner


def head_split(values: Mapping[int, torch.Tensor], layers, num_heads):
    """{layer: [positions, hidden]} -> [n_layers, num_heads, head_dim] float16."""
    stacked = []
    for layer in layers:
        tensor = values.get(layer)
        if tensor is None or tensor.numel() == 0:
            return None
        vector = tensor[-1].float()  # position="last" yields one row
        if vector.numel() % num_heads:
            raise ValueError(
                f"hidden {vector.numel()} not divisible by num_heads {num_heads}"
            )
        stacked.append(vector.view(num_heads, -1))
    return torch.stack(stacked).to(torch.float16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--model-family", default="qwen3_vl")
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--input-data", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--retrieval-manifest", type=Path, required=True)
    parser.add_argument("--retrieval-image-root", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mc-prompt-key", default="MC_1")
    parser.add_argument("--sa-prompt-key", default="SA_1")
    parser.add_argument("--la-prompt-key", default="LA_1")
    parser.add_argument("--prompts", type=Path, default=None)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--canonical-size", default="672,672")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=17)
    parser.add_argument(
        "--conditions", default=",".join(CONDITIONS), help="Comma-separated subset"
    )
    parser.add_argument(
        "--skip-activations",
        action="store_true",
        help="Score only; skip the activation dump (phase A go/no-go)",
    )
    args = parser.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise SystemExit(f"Unknown conditions {unknown}; choose from {list(CONDITIONS)}")
    canonical = tuple(int(v) for v in args.canonical_size.split(",")) if args.canonical_size else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_prompt_overrides(args.prompts)
    runner = build_runner(args)

    manifest_rows = read_csv(args.retrieval_manifest)
    index = retrieval_index(manifest_rows, args.split, args.retrieval_image_root)
    all_candidates = text_index(manifest_rows, args.split)
    rows = [r for r in runner.rows_filtered(["MC"], ["full"], args.limit)]
    print(
        f"MC 문항 {len(rows)}개 | 검색 이미지 확보 {len(index)}개 | "
        f"텍스트 근거 확보 {len(all_candidates)}개",
        flush=True,
    )

    # Donor for orig_shuffled: another question's rank-1 retrieved image.
    question_ids = sorted(index)
    rng = random.Random(args.shuffle_seed)
    donors: dict[str, Path] = {}
    if len(question_ids) > 1:
        rotated = question_ids[1:] + question_ids[:1]
        rng.shuffle(rotated)
        for qid, donor in zip(question_ids, rotated):
            if donor == qid:
                donor = question_ids[(question_ids.index(qid) + 1) % len(question_ids)]
            donors[qid] = index[donor][0]["path"]

    layers = (
        list(range(runner.n_layers))
        if args.layers == "all"
        else [int(v) for v in args.layers.split(",")]
    )
    writer = R.ResultWriter(args.output_dir, "01_retrieval_eval")
    cache_dir = args.output_dir / "01_retrieval_eval" / "activation_cache"
    if not args.skip_activations:
        cache_dir.mkdir(parents=True, exist_ok=True)

    n_covered = 0
    for row in tqdm(rows, desc="retrieval_eval"):
        qid = str(row["question_id"])
        candidates = index.get(qid, [])
        if candidates:
            n_covered += 1
        payload: dict[str, Any] = {"question_id": qid, "layers": layers, "activations": {}}
        cache_path = cache_dir / f"{qid}.pt"
        if not args.skip_activations and cache_path.is_file():
            # Resume must consider activations, not just scores: a prior
            # --skip-activations run already wrote the result rows, and skipping
            # on those alone would silently leave the cache empty.
            try:
                payload["activations"] = dict(
                    R.torch.load(cache_path, map_location="cpu").get("activations", {})
                )
            except Exception:
                payload["activations"] = {}

        for condition in conditions:
            variant = dict(row)
            variant["_target_image_size"] = list(canonical) if canonical else None
            if condition == "text_only":
                variant["input_image_variant"] = "none"
                variant["_extra_images"] = []
            elif condition == "orig":
                variant["_extra_images"] = []
            elif condition == "orig_top1":
                if not candidates:
                    continue
                variant["_extra_images"] = [candidates[0]["path"]]
            elif condition == "orig_shuffled":
                if qid not in donors:
                    continue
                variant["_extra_images"] = [donors[qid]]
            elif condition == "ret_only":
                if not candidates:
                    continue
                variant["input_image_variant"] = "none"
                variant["_extra_images"] = [candidates[0]["path"]]
            elif condition == "orig_top1_labeled":
                if not candidates:
                    continue
                variant["_extra_images"] = [candidates[0]["path"]]
                variant["_prompt_prefix"] = IMAGE_NOTICE
            elif condition == "orig_plus_text":
                text_block = evidence_text(all_candidates.get(qid, []))
                if not text_block:
                    continue
                variant["_extra_images"] = []
                variant["_prompt_prefix"] = text_block
            elif condition == "orig_plus_both":
                text_block = evidence_text(all_candidates.get(qid, []))
                if not candidates or not text_block:
                    continue
                variant["_extra_images"] = [candidates[0]["path"]]
                variant["_prompt_prefix"] = IMAGE_NOTICE + text_block
            elif condition.startswith("orig_top1_v"):
                if not candidates:
                    continue
                spec = NOTICE_VARIANTS[condition.rsplit("_", 1)[-1]]
                variant["_extra_images"] = [candidates[0]["path"]]
                variant["_prompt_prefix"] = spec["prefix"]
                if spec["markers"]:
                    variant["_image_captions"] = list(spec["markers"])
            variant["sample_uid"] = f"{row['sample_uid']}::{condition}"

            result_id = f"{qid}::{condition}"
            have_score = writer.has(result_id)
            have_activation = args.skip_activations or condition in payload["activations"]
            if have_score and have_activation:
                continue

            if args.skip_activations:
                score = runner.score_mc(variant, cache_baseline=False)
            else:
                capture: dict[str, Any] = {}

                def factory(prepared, _capture=capture):
                    hook = R.MultiLayerHeadCapture(runner.model, prepared, layers, "last")
                    _capture["hook"] = hook
                    return hook

                score = runner.score_mc(variant, factory, cache_baseline=False)
                hook = capture.get("hook")
                if hook is not None:
                    split = head_split(hook.values, layers, runner.num_heads)
                    if split is not None:
                        payload["activations"][condition] = split

            top1 = candidates[0] if candidates else {}
            writer.append(
                {
                    **score,
                    "result_id": result_id,
                    "experiment": "01_retrieval_eval",
                    "experiment_type": "retrieval_eval",
                    "question_id": qid,
                    "question_form": "MC",
                    "condition": condition,
                    "reference": row.get("reference", ""),
                    "n_candidates": len(candidates),
                    "top1_similarity": top1.get("similarity", ""),
                    "top1_title": top1.get("title", ""),
                    "top1_description": top1.get("description", ""),
                }
            )

        if not args.skip_activations and payload["activations"]:
            torch.save(payload, cache_path)

    writer.finalize()
    manifest = {
        "split": args.split,
        "conditions": conditions,
        "n_rows": len(rows),
        "n_with_retrieved_image": n_covered,
        "layers": layers,
        "num_heads": runner.num_heads,
        "canonical_size": list(canonical) if canonical else None,
        "shuffle_seed": args.shuffle_seed,
        "activations_saved": not args.skip_activations,
        "cache_dir": str(cache_dir),
    }
    (args.output_dir / "01_retrieval_eval" / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n완료: {args.output_dir / '01_retrieval_eval'}")
    print(f"  검색 이미지 확보: {n_covered}/{len(rows)}")


if __name__ == "__main__":
    main()
