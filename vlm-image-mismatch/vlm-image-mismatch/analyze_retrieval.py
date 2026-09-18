#!/usr/bin/env python3
"""Retrieval-augmented analysis: go/no-go, then the gated policy.

Phase A (scores only)
  Accuracy per condition, paired against `orig`, with exact McNemar and a
  similarity-stratified breakdown.  If orig_top1 < orig, naive attachment is a
  net loss and a gate is required for the direction to be worth anything.

Phase B (needs activations)
  Fit a two-image alignment probe on {orig_top1 = aligned, orig_shuffled =
  mismatched} using condition labels only, then simulate the deployment policy:
  attach the retrieved image when the probe says aligned, otherwise keep `orig`.
  Reported against the naive and oracle policies so the gate's contribution is
  visible on its own.

Correctness labels never enter probe fitting; they are used only to score the
policies.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from analyze_mismatch_activations import (
    binary_auc,
    average_precision,
    cross_validated_scores,
    exact_mcnemar,
    fit_logistic,
    logistic_scores,
    load_pt,
    threshold_at_fpr,
    write_csv,
    youden_threshold,
)


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def read_scores(results_csv: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """{question_id: {condition: row}}"""
    table: dict[str, dict[str, dict[str, Any]]] = {}
    with io.open(results_csv, encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            table.setdefault(str(row["question_id"]), {})[str(row["condition"])] = {
                "correct": truthy(row.get("is_correct")),
                "margin": as_float(row.get("gold_margin")),
                "similarity": as_float(row.get("top1_similarity")),
                "n_candidates": as_float(row.get("n_candidates")) or 0,
            }
    return table


def paired(table, reference: str, condition: str) -> dict[str, Any]:
    shared = [q for q, v in table.items() if reference in v and condition in v]
    if not shared:
        return {}
    before = [table[q][reference]["correct"] for q in shared]
    after = [table[q][condition]["correct"] for q in shared]
    repairs = sum((not b) and a for b, a in zip(before, after))
    damage = sum(b and (not a) for b, a in zip(before, after))
    return {
        "condition": condition,
        "n": len(shared),
        "reference_accuracy": sum(before) / len(shared),
        "accuracy": sum(after) / len(shared),
        "accuracy_gain": (sum(after) - sum(before)) / len(shared),
        "repairs": repairs,
        "damage": damage,
        "net_repairs": repairs - damage,
        "mcnemar_exact_two_sided_p": exact_mcnemar(repairs, damage),
    }


def phase_a(table, reference: str, output_dir: Path) -> list[dict[str, Any]]:
    conditions = sorted({c for v in table.values() for c in v})
    rows = [paired(table, reference, c) for c in conditions]
    rows = [r for r in rows if r]
    write_csv(output_dir / "retrieval_conditions.csv", rows)

    # Does retrieval quality predict whether attaching helps?
    strata: list[dict[str, Any]] = []
    scored = [
        (q, v)
        for q, v in table.items()
        if "orig" in v and "orig_top1" in v and v["orig_top1"]["similarity"] is not None
    ]
    if scored:
        scored.sort(key=lambda item: item[1]["orig_top1"]["similarity"])
        buckets = 4
        size = max(1, len(scored) // buckets)
        for index in range(buckets):
            chunk = scored[index * size : (index + 1) * size if index < buckets - 1 else None]
            if not chunk:
                continue
            before = [v["orig"]["correct"] for _q, v in chunk]
            after = [v["orig_top1"]["correct"] for _q, v in chunk]
            sims = [v["orig_top1"]["similarity"] for _q, v in chunk]
            strata.append(
                {
                    "similarity_quartile": index + 1,
                    "similarity_min": min(sims),
                    "similarity_max": max(sims),
                    "n": len(chunk),
                    "orig_accuracy": sum(before) / len(chunk),
                    "orig_top1_accuracy": sum(after) / len(chunk),
                    "gain": (sum(after) - sum(before)) / len(chunk),
                }
            )
        write_csv(output_dir / "retrieval_similarity_strata.csv", strata)
    return rows


# ------------------------------------------------------------------- phase B


def load_activations(cache_dir: Path, conditions: Sequence[str]):
    samples: list[dict[str, Any]] = []
    for path in sorted(cache_dir.glob("*.pt")):
        payload = load_pt(path)
        available = payload.get("activations", {})
        for condition in conditions:
            tensor = available.get(condition)
            if tensor is None:
                continue
            samples.append(
                {
                    "question_id": str(payload["question_id"]),
                    "condition": condition,
                    "activation": tensor,
                }
            )
    return samples


def features_for(samples, heads: Sequence[Sequence[int]], layers: Sequence[int]):
    layer_pos = {layer: i for i, layer in enumerate(layers)}
    out = []
    for sample in samples:
        vectors = [
            sample["activation"][layer_pos[int(l)], int(h)].float().flatten()
            for l, h in heads
        ]
        out.append(torch.cat(vectors).tolist())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", default="orig")
    parser.add_argument("--target", default="orig_top1")
    parser.add_argument("--mismatch-condition", default="orig_shuffled")
    parser.add_argument("--layers", default=None, help="e.g. 48,53,55,60,63")
    parser.add_argument("--heads-per-layer", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--target-fpr", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment = args.eval_dir / "01_retrieval_eval"
    table = read_scores(experiment / "results.csv")
    print(f"문항 {len(table)}개\n")

    rows = phase_a(table, args.reference, args.output_dir)
    print("=== Phase A: 조건별 정확도 (reference =", args.reference, ") ===")
    for row in rows:
        print(
            f"  {row['condition']:<14} acc {row['accuracy']:.4f}  "
            f"gain {row['accuracy_gain']:+.4f}  "
            f"rep/dam {row['repairs']}/{row['damage']}  p={row['mcnemar_exact_two_sided_p']:.4f}"
        )
    target = next((r for r in rows if r["condition"] == args.target), None)
    if target:
        verdict = (
            "naive 붙이기가 이득 -> 게이트는 추가 이득용"
            if target["accuracy_gain"] > 0
            else "naive 붙이기가 손해 -> 게이트 없이는 이 방향이 마이너스"
        )
        print(f"\n  판정: {verdict}")

    manifest_path = experiment / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    cache_dir = Path(manifest.get("cache_dir", experiment / "activation_cache"))
    if not manifest.get("activations_saved") or not cache_dir.is_dir():
        print("\nactivation이 없어 Phase B는 건너뜁니다 (--skip-activations로 돌린 실행).")
        return

    layers = manifest.get("layers") or []
    num_heads = int(manifest.get("num_heads", 0))
    selected_layers = (
        [int(v) for v in args.layers.split(",")]
        if args.layers
        else layers[-5:]  # late layers carry the alignment signal
    )
    heads = [
        (layer, head)
        for layer in selected_layers
        for head in range(min(args.heads_per_layer, num_heads))
    ]

    samples = load_activations(cache_dir, [args.target, args.mismatch_condition])
    present = {s["condition"] for s in samples}
    missing = [c for c in (args.target, args.mismatch_condition) if c not in present]
    if missing:
        # The probe needs both classes. A --skip-activations run scores the
        # condition but stores nothing, so this is the usual cause.
        print(
            f"\nPhase B 건너뜀: activation이 없는 조건 {missing}.\n"
            f"  캐시에 있는 조건: {sorted(present) or '없음'}\n"
            "  Phase B가 필요하면 --skip-activations 없이 해당 조건을 다시 실행하세요.\n"
            "  (텍스트 근거 조건은 두 번째 이미지가 없어 2-image probe 대상이 아닙니다.)"
        )
        return
    labels = [s["condition"] == args.mismatch_condition for s in samples]
    groups = [s["question_id"] for s in samples]
    features = features_for(samples, heads, layers)

    cv = cross_validated_scores(features, labels, groups, args.folds, args.seed, args.l2)
    auroc = binary_auc(cv, labels)
    print(f"\n=== Phase B: 2-image alignment probe ===")
    print(f"  heads={len(heads)}  samples={len(samples)}  group-CV AUROC={auroc:.4f}")
    print(f"  AUPRC={average_precision(cv, labels):.4f}")

    model = fit_logistic(features, labels, args.l2)
    scores = logistic_scores(model, features)
    thresholds = {
        "youden": youden_threshold(scores, labels),
        f"fpr_{args.target_fpr}": threshold_at_fpr(scores, labels, args.target_fpr),
    }

    # Deployment simulation: keep `orig` when the probe flags the pair as
    # mismatched, otherwise use the attached-retrieval answer.
    by_question = {}
    for sample, score in zip(samples, scores):
        if sample["condition"] == args.target:
            by_question[sample["question_id"]] = score

    policies: list[dict[str, Any]] = []
    shared = [q for q in by_question if q in table and args.target in table[q]]
    ref_correct = [table[q][args.reference]["correct"] for q in shared]
    tgt_correct = [table[q][args.target]["correct"] for q in shared]
    policies.append(
        {
            "policy": f"always {args.reference}",
            "n": len(shared),
            "accuracy": sum(ref_correct) / len(shared),
        }
    )
    policies.append(
        {
            "policy": f"always {args.target} (naive)",
            "n": len(shared),
            "accuracy": sum(tgt_correct) / len(shared),
        }
    )
    for name, threshold in thresholds.items():
        final = [
            t if by_question[q] < threshold else r
            for q, r, t in zip(shared, ref_correct, tgt_correct)
        ]
        repairs = sum((not r) and f for r, f in zip(ref_correct, final))
        damage = sum(r and (not f) for r, f in zip(ref_correct, final))
        policies.append(
            {
                "policy": f"gated ({name})",
                "n": len(shared),
                "attach_rate": sum(by_question[q] < threshold for q in shared) / len(shared),
                "accuracy": sum(final) / len(shared),
                "gain_vs_reference": (sum(final) - sum(ref_correct)) / len(shared),
                "repairs": repairs,
                "damage": damage,
                "mcnemar_exact_two_sided_p": exact_mcnemar(repairs, damage),
            }
        )
    oracle = [r or t for r, t in zip(ref_correct, tgt_correct)]
    policies.append(
        {
            "policy": "oracle (upper bound)",
            "n": len(shared),
            "accuracy": sum(oracle) / len(shared),
            "gain_vs_reference": (sum(oracle) - sum(ref_correct)) / len(shared),
        }
    )
    write_csv(args.output_dir / "retrieval_policies.csv", policies)

    # Frozen artifact for submit_test.py. Attach the retrieved image when the
    # probe score is BELOW the threshold (probe predicts "mismatched").
    (args.output_dir / "retrieval_probe.json").write_text(
        json.dumps(
            {
                "model": model,
                "heads": [[int(l), int(h)] for l, h in heads],
                "layers": [int(v) for v in layers],
                "num_heads": num_heads,
                "thresholds": thresholds,
                "default_threshold_policy": "youden",
                "target_condition": args.target,
                "mismatch_condition": args.mismatch_condition,
                "attach_if": "score < threshold",
                "cv_auroc": auroc,
                "uses_correctness_labels": False,
            },
            ensure_ascii=False,
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    print(f"  probe artifact: {args.output_dir / 'retrieval_probe.json'}")
    print("\n=== 정책 비교 ===")
    for row in policies:
        extra = (
            f"  attach={row['attach_rate']:.2f}" if "attach_rate" in row else ""
        )
        print(f"  {row['policy']:<26} acc {row['accuracy']:.4f}{extra}")

    (args.output_dir / "retrieval_report.json").write_text(
        json.dumps(
            {
                "phase_a": rows,
                "probe_auroc": auroc,
                "n_heads": len(heads),
                "selected_layers": selected_layers,
                "thresholds": thresholds,
                "policies": policies,
                "probe_uses_correctness_labels": False,
            },
            ensure_ascii=False,
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    print(f"\n완료: {args.output_dir}")


if __name__ == "__main__":
    main()
