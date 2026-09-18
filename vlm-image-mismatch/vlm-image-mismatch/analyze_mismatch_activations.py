#!/usr/bin/env python3
"""CPU analysis for length-matched VLM mismatch activations.

Calibration discovers heads and fits condition-label probes on Train only.
Evaluation loads the frozen artifact and reports AUROC/AP, incremental AUROC over
margin+entropy, and Text-only fallback utility.  Correctness labels never enter
head selection or probe fitting; they are used only for routing evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            cooked: dict[str, Any] = {}
            for key in fieldnames:
                value = jsonable(row.get(key, ""))
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)
                cooked[key] = value
            writer.writerow(cooked)


def load_pt(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_export(export_dir: Path) -> tuple[dict[str, Any], list[Path]]:
    manifest_path = export_dir / "export_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing export manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    configured_cache = Path(str(manifest.get("cache_dir", "")))
    if configured_cache.is_dir():
        cache_dir = configured_cache
    else:
        candidates = sorted(export_dir.glob("activation_cache_*"))
        if not candidates:
            raise FileNotFoundError(f"No activation cache under {export_dir}")
        cache_dir = candidates[-1]
    files = sorted(cache_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No activation shards in {cache_dir}")
    expected = int(manifest.get("n_questions", len(files)))
    if len(files) != expected:
        raise RuntimeError(
            f"Activation export is incomplete: {len(files)}/{expected} files"
        )
    return manifest, files


def binary_auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    positives = sum(bool(label) for label in labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(
        zip((float(score) for score in scores), labels), key=lambda item: item[0]
    )
    positive_rank_sum = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        positive_rank_sum += average_rank * sum(
            bool(label) for _score, label in ordered[start:end]
        )
        start = end
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def average_precision(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    if not any(labels):
        return None
    ranked = sorted(
        zip((float(value) for value in scores), labels),
        key=lambda item: item[0],
        reverse=True,
    )
    hits = 0
    total = 0.0
    for rank, (_score, label) in enumerate(ranked, start=1):
        if label:
            hits += 1
            total += hits / rank
    return total / hits


def pearson(a: Sequence[float], b: Sequence[float]) -> float | None:
    if len(a) != len(b) or len(a) < 2:
        return None
    x = torch.tensor(list(a), dtype=torch.float64)
    y = torch.tensor(list(b), dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.sqrt(x.square().sum() * y.square().sum())
    if float(denominator.item()) <= 0.0:
        return None
    return float((x @ y / denominator).item())


def tied_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: float(values[index]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and float(values[ordered[end]]) == float(
            values[ordered[start]]
        ):
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        for position in range(start, end):
            ranks[ordered[position]] = rank
        start = end
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    return pearson(tied_ranks(a), tied_ranks(b))


def zscore(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-8)


def safe_metric_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return float(value - baseline)


def discover_heads(
    files: Sequence[Path],
    manifest: Mapping[str, Any],
    output_dir: Path,
    top_k: int,
    max_heads_per_layer: int,
    random_head_sets: int,
    seed: int,
) -> tuple[dict[str, list[list[int]]], dict[str, Any]]:
    totals: dict[str, torch.Tensor] = {}
    count = 0
    layer_indices: list[int] | None = None
    num_heads = int(manifest["num_heads"])
    for path in files:
        payload = load_pt(path)
        layers = [int(value) for value in payload["layers"]]
        if layer_indices is None:
            layer_indices = layers
        elif layers != layer_indices:
            raise RuntimeError("Activation shards contain inconsistent layer lists")
        distances = payload["distances"]
        matrices = {
            "full_blurred_l2": distances["full_blurred"]["symmetric_l2"],
            "full_blurred_cosine": distances["full_blurred"]["cosine"],
            "full_shuffled_l2": distances["full_shuffled"]["symmetric_l2"],
            "full_shuffled_cosine": distances["full_shuffled"]["cosine"],
            "full_norm": distances["full_blurred"]["source_norm"],
            "blurred_norm": distances["full_blurred"]["target_norm"],
            "shuffled_norm": distances["full_shuffled"]["target_norm"],
            "option_permutation": distances.get(
                "option_permutation",
                torch.zeros_like(distances["full_blurred"]["symmetric_l2"]),
            ),
        }
        if "full_text" in distances:
            matrices["full_text_l2"] = distances["full_text"]["symmetric_l2"]
            matrices["text_norm"] = distances["full_text"]["target_norm"]
        for name, matrix in matrices.items():
            value = matrix.float()
            totals[name] = totals.get(name, torch.zeros_like(value)) + value
        count += 1
    if not count or layer_indices is None:
        raise RuntimeError("No activation payloads were available for discovery")
    means = {name: value / count for name, value in totals.items()}
    fb = means["full_blurred_l2"]
    fs = means["full_shuffled_l2"]
    permutation = means["option_permutation"]
    visual_discounted = fb / (1.0 + permutation)
    mismatch_specificity = fs - fb
    mismatch_discounted = mismatch_specificity / (1.0 + permutation)
    full_text = means.get("full_text_l2")

    rows: list[dict[str, Any]] = []
    for layer_position, layer_idx in enumerate(layer_indices):
        layer_visual_z = zscore(visual_discounted[layer_position])
        layer_mismatch_z = zscore(mismatch_discounted[layer_position])
        for head_idx in range(num_heads):
            rows.append(
                {
                    "layer": layer_idx,
                    "head": head_idx,
                    "n_questions": count,
                    "mean_full_blurred_symmetric_l2": float(
                        fb[layer_position, head_idx].item()
                    ),
                    "mean_full_shuffled_symmetric_l2": float(
                        fs[layer_position, head_idx].item()
                    ),
                    "mean_full_text_symmetric_l2": (
                        float(full_text[layer_position, head_idx].item())
                        if full_text is not None
                        else ""
                    ),
                    "mean_full_blurred_cosine": float(
                        means["full_blurred_cosine"][layer_position, head_idx].item()
                    ),
                    "mean_full_shuffled_cosine": float(
                        means["full_shuffled_cosine"][layer_position, head_idx].item()
                    ),
                    "mean_option_permutation_sensitivity": float(
                        permutation[layer_position, head_idx].item()
                    ),
                    "visual_score_discounted": float(
                        visual_discounted[layer_position, head_idx].item()
                    ),
                    "mismatch_specificity": float(
                        mismatch_specificity[layer_position, head_idx].item()
                    ),
                    "mismatch_score_discounted": float(
                        mismatch_discounted[layer_position, head_idx].item()
                    ),
                    "visual_within_layer_z": float(
                        layer_visual_z[head_idx].item()
                    ),
                    "mismatch_within_layer_z": float(
                        layer_mismatch_z[head_idx].item()
                    ),
                    "mean_full_norm": float(
                        means["full_norm"][layer_position, head_idx].item()
                    ),
                    "mean_blurred_norm": float(
                        means["blurred_norm"][layer_position, head_idx].item()
                    ),
                    "mean_shuffled_norm": float(
                        means["shuffled_norm"][layer_position, head_idx].item()
                    ),
                    "mean_text_norm": (
                        float(means["text_norm"][layer_position, head_idx].item())
                        if "text_norm" in means
                        else ""
                    ),
                }
            )

    def select(metric: str) -> list[list[int]]:
        selected: list[list[int]] = []
        per_layer: dict[int, int] = {}
        for row in sorted(rows, key=lambda item: float(item[metric]), reverse=True):
            layer = int(row["layer"])
            if per_layer.get(layer, 0) >= max_heads_per_layer:
                continue
            selected.append([layer, int(row["head"])])
            per_layer[layer] = per_layer.get(layer, 0) + 1
            if len(selected) >= top_k:
                break
        return selected

    available = {(int(row["layer"]), int(row["head"])) for row in rows}
    legacy = [
        [int(layer), int(head)]
        for layer, head in manifest.get("legacy_heads", [])
        if (int(layer), int(head)) in available
    ]
    selected_groups = {
        "legacy": legacy,
        "visual": select("visual_score_discounted"),
        "mismatch": select("mismatch_score_discounted"),
    }
    excluded = {
        (int(layer), int(head))
        for group in ("legacy", "visual", "mismatch")
        for layer, head in selected_groups[group]
    }
    available_random = sorted(available - excluded)
    if len(available_random) < top_k:
        available_random = sorted(available)
    for random_index in range(random_head_sets):
        rng = random.Random(f"{seed}:random_head_set:{random_index}")
        selected_groups[f"random_{random_index + 1:02d}"] = [
            [layer, head] for layer, head in rng.sample(available_random, top_k)
        ]
    selected_lookup: dict[tuple[int, int], list[str]] = {}
    for group, values in selected_groups.items():
        for layer, head in values:
            selected_lookup.setdefault((int(layer), int(head)), []).append(group)
    for row in rows:
        row["selected_groups"] = sorted(
            selected_lookup.get((int(row["layer"]), int(row["head"])), [])
        )
    for rank_name, metric_name in (
        ("visual_rank", "visual_score_discounted"),
        ("mismatch_rank", "mismatch_score_discounted"),
        ("full_text_rank", "mean_full_text_symmetric_l2"),
    ):
        eligible = [
            row
            for row in rows
            if row[metric_name] != ""
        ]
        for rank, row in enumerate(
            sorted(eligible, key=lambda item: float(item[metric_name]), reverse=True),
            start=1,
        ):
            row[rank_name] = rank
    rows.sort(key=lambda item: float(item["mismatch_score_discounted"]), reverse=True)
    write_csv(output_dir / "head_discovery.csv", rows)
    write_csv(
        output_dir / "legacy_head_diagnostics.csv",
        [
            row
            for row in rows
            if [int(row["layer"]), int(row["head"])] in legacy
        ],
    )

    ft_values = full_text.flatten().tolist() if full_text is not None else []
    fb_values = fb.flatten().tolist()
    fs_values = fs.flatten().tolist()
    summary = {
        "n_questions": count,
        "layers": layer_indices,
        "num_heads": num_heads,
        "selected_head_groups": selected_groups,
        "spearman_full_text_vs_full_blurred": (
            spearman(ft_values, fb_values) if ft_values else None
        ),
        "spearman_full_text_vs_full_shuffled": (
            spearman(ft_values, fs_values) if ft_values else None
        ),
        "spearman_full_blurred_vs_full_shuffled": spearman(fb_values, fs_values),
        "selection_uses_correctness_labels": False,
        "visual_selection_metric": "mean_d(Full,Blurred)/(1+option_permutation_sensitivity)",
        "mismatch_selection_metric": "(mean_d(Full,Shuffled)-mean_d(Full,Blurred))/(1+option_permutation_sensitivity)",
    }
    (output_dir / "discovery_summary.json").write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return selected_groups, summary


def head_features(
    activation: torch.Tensor,
    layer_to_position: Mapping[int, int],
    heads: Sequence[Sequence[int]],
) -> list[float]:
    vectors = [
        activation[layer_to_position[int(layer)], int(head)].float().flatten()
        for layer, head in heads
    ]
    if not vectors:
        return []
    return torch.cat(vectors).tolist()


def build_samples(
    files: Sequence[Path], selected_groups: Mapping[str, Sequence[Sequence[int]]]
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for path in files:
        payload = load_pt(path)
        question_id = str(payload["question_id"])
        layers = [int(value) for value in payload["layers"]]
        layer_to_position = {layer: index for index, layer in enumerate(layers)}
        text_score = payload["scores"]["text_only"]
        for condition in ("full", "blurred", "shuffled"):
            score = payload["scores"][condition]
            activation = payload["activations"][condition]
            sample: dict[str, Any] = {
                "question_id": question_id,
                "condition": condition,
                "mismatch": condition == "shuffled",
                "uncertainty": [
                    -float(score["top1_margin"]),
                    float(score["candidate_entropy"]),
                ],
                "baseline_correct": score.get("is_correct"),
                "fallback_correct": text_score.get("is_correct"),
                "baseline_prediction": score.get("prediction"),
                "fallback_prediction": text_score.get("prediction"),
            }
            for group, heads in selected_groups.items():
                if heads:
                    sample[f"activation_{group}"] = head_features(
                        activation, layer_to_position, heads
                    )
            samples.append(sample)
    return samples


def fit_logistic(
    features: Sequence[Sequence[float]],
    labels: Sequence[bool],
    l2: float,
) -> dict[str, Any]:
    if not features or not any(labels) or all(labels):
        raise ValueError("Logistic fitting requires both mismatch classes")
    x = torch.tensor(list(features), dtype=torch.float64)
    y = torch.tensor([float(value) for value in labels], dtype=torch.float64)
    means = x.mean(dim=0)
    scales = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    z = (x - means) / scales
    weights = torch.zeros(z.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weights, bias], lr=0.5, max_iter=200, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = z @ weights + bias
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss = loss + float(l2) * weights.square().sum() / len(labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return {
        "means": means.tolist(),
        "scales": scales.tolist(),
        "weights": weights.detach().tolist(),
        "bias": float(bias.detach().item()),
        "l2": float(l2),
        "feature_dim": int(x.shape[1]),
    }


def logistic_scores(
    model: Mapping[str, Any], features: Sequence[Sequence[float]]
) -> list[float]:
    means = [float(value) for value in model["means"]]
    scales = [float(value) for value in model["scales"]]
    weights = [float(value) for value in model["weights"]]
    bias = float(model["bias"])
    result: list[float] = []
    for values in features:
        logit = bias + sum(
            weight * ((float(value) - mean) / scale)
            for value, mean, scale, weight in zip(values, means, scales, weights)
        )
        if logit >= 0:
            result.append(1.0 / (1.0 + math.exp(-logit)))
        else:
            exponential = math.exp(logit)
            result.append(exponential / (1.0 + exponential))
    return result


def feature_sets(samples: Sequence[Mapping[str, Any]]) -> dict[str, list[list[float]]]:
    sets: dict[str, list[list[float]]] = {
        "uncertainty": [list(sample["uncertainty"]) for sample in samples]
    }
    group_names = sorted(
        {
            key.removeprefix("activation_")
            for sample in samples
            for key in sample
            if key.startswith("activation_")
        }
    )
    for group in group_names:
        activation = [list(sample[f"activation_{group}"]) for sample in samples]
        sets[f"activation_{group}"] = activation
        sets[f"uncertainty_plus_activation_{group}"] = [
            list(sample["uncertainty"]) + values
            for sample, values in zip(samples, activation)
        ]
    return sets


def group_folds(groups: Sequence[str], folds: int, seed: int) -> list[int]:
    unique = sorted(
        set(groups),
        key=lambda value: hashlib.sha1(f"{seed}:{value}".encode()).hexdigest(),
    )
    assignment = {group: index % folds for index, group in enumerate(unique)}
    return [assignment[group] for group in groups]


def cross_validated_scores(
    features: Sequence[Sequence[float]],
    labels: Sequence[bool],
    groups: Sequence[str],
    folds: int,
    seed: int,
    l2: float,
) -> list[float]:
    assignments = group_folds(groups, folds, seed)
    scores = [float("nan")] * len(labels)
    for fold in range(folds):
        train = [index for index, value in enumerate(assignments) if value != fold]
        valid = [index for index, value in enumerate(assignments) if value == fold]
        model = fit_logistic(
            [features[index] for index in train],
            [labels[index] for index in train],
            l2,
        )
        predicted = logistic_scores(model, [features[index] for index in valid])
        for index, value in zip(valid, predicted):
            scores[index] = value
    if any(not math.isfinite(value) for value in scores):
        raise RuntimeError("Cross-validation failed to score every sample")
    return scores


def model_metrics(
    name: str,
    scores: Sequence[float],
    samples: Sequence[Mapping[str, Any]],
    baseline_auc: float | None,
) -> dict[str, Any]:
    labels = [bool(sample["mismatch"]) for sample in samples]
    conditions = [str(sample["condition"]) for sample in samples]
    auc_all = binary_auc(scores, labels)
    full_shuffled = [
        index
        for index, condition in enumerate(conditions)
        if condition in {"full", "shuffled"}
    ]
    subset_scores = [scores[index] for index in full_shuffled]
    subset_labels = [labels[index] for index in full_shuffled]
    clipped = [min(1.0 - 1e-9, max(1e-9, float(value))) for value in scores]
    log_loss = -sum(
        math.log(probability if label else 1.0 - probability)
        for probability, label in zip(clipped, labels)
    ) / len(labels)
    brier = sum(
        (probability - float(label)) ** 2
        for probability, label in zip(clipped, labels)
    ) / len(labels)
    margin_risk = [float(sample["uncertainty"][0]) for sample in samples]
    entropy = [float(sample["uncertainty"][1]) for sample in samples]
    return {
        "model": name,
        "n": len(samples),
        "n_questions": len({str(sample["question_id"]) for sample in samples}),
        "n_mismatch": sum(labels),
        "auroc_shuffled_vs_full_blurred": auc_all,
        "ap_shuffled_vs_full_blurred": average_precision(scores, labels),
        "auroc_shuffled_vs_full": binary_auc(subset_scores, subset_labels),
        "delta_auroc_over_uncertainty": safe_metric_delta(auc_all, baseline_auc),
        "log_loss": log_loss,
        "brier": brier,
        "pearson_with_negative_margin": pearson(scores, margin_risk),
        "pearson_with_entropy": pearson(scores, entropy),
    }


def rates_at_threshold(
    scores: Sequence[float], labels: Sequence[bool], threshold: float
) -> tuple[float, float]:
    predicted = [float(value) >= threshold for value in scores]
    positives = sum(labels)
    negatives = len(labels) - positives
    tpr = (
        sum(prediction and label for prediction, label in zip(predicted, labels))
        / positives
        if positives
        else 0.0
    )
    fpr = (
        sum(prediction and not label for prediction, label in zip(predicted, labels))
        / negatives
        if negatives
        else 0.0
    )
    return tpr, fpr


def threshold_at_fpr(
    scores: Sequence[float], labels: Sequence[bool], target_fpr: float
) -> float:
    candidates = [float("inf")] + sorted(
        {float(value) for value in scores}, reverse=True
    )
    feasible: list[tuple[float, float, float]] = []
    for threshold in candidates:
        tpr, fpr = rates_at_threshold(scores, labels, threshold)
        if fpr <= target_fpr + 1e-12:
            feasible.append((tpr, fpr, threshold))
    if not feasible:
        return float("inf")
    return max(feasible, key=lambda item: (item[0], item[1], -item[2]))[2]


def youden_threshold(scores: Sequence[float], labels: Sequence[bool]) -> float:
    candidates = [float("inf")] + sorted(
        {float(value) for value in scores}, reverse=True
    )
    return max(
        candidates,
        key=lambda threshold: (
            rates_at_threshold(scores, labels, threshold)[0]
            - rates_at_threshold(scores, labels, threshold)[1]
        ),
    )


def exact_mcnemar(repairs: int, damage: int) -> float:
    discordant = repairs + damage
    if discordant == 0:
        return 1.0
    smaller = min(repairs, damage)
    tail = sum(math.comb(discordant, index) for index in range(smaller + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def routing_rows(
    model_name: str,
    scores: Sequence[float],
    thresholds: Mapping[str, float],
    samples: Sequence[Mapping[str, Any]],
    mixture_prevalences: Sequence[float],
    aligned_full_weight: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for policy_name, threshold in thresholds.items():
        gates = [float(value) >= float(threshold) for value in scores]
        condition_metrics: dict[str, dict[str, float]] = {}
        for condition in ("full", "blurred", "shuffled"):
            indices = [
                index
                for index, sample in enumerate(samples)
                if sample["condition"] == condition
            ]
            baseline = [bool(samples[index]["baseline_correct"]) for index in indices]
            final = [
                bool(samples[index]["fallback_correct"])
                if gates[index]
                else bool(samples[index]["baseline_correct"])
                for index in indices
            ]
            repairs = sum((not before) and after for before, after in zip(baseline, final))
            damage = sum(before and (not after) for before, after in zip(baseline, final))
            gate_count = sum(gates[index] for index in indices)
            baseline_accuracy = sum(baseline) / len(indices)
            routed_accuracy = sum(final) / len(indices)
            condition_metrics[condition] = {
                "baseline_accuracy": baseline_accuracy,
                "routed_accuracy": routed_accuracy,
            }
            rows.append(
                {
                    "model": model_name,
                    "policy": policy_name,
                    "threshold": threshold,
                    "condition": condition,
                    "mixture_mismatch_prevalence": "",
                    "n": len(indices),
                    "gate_count": gate_count,
                    "gate_rate": gate_count / len(indices),
                    "baseline_accuracy": baseline_accuracy,
                    "routed_accuracy": routed_accuracy,
                    "accuracy_gain": routed_accuracy - baseline_accuracy,
                    "repairs": repairs,
                    "damage": damage,
                    "net_repairs": repairs - damage,
                    "mcnemar_exact_two_sided_p": exact_mcnemar(repairs, damage),
                }
            )
        aligned_baseline = (
            aligned_full_weight * condition_metrics["full"]["baseline_accuracy"]
            + (1.0 - aligned_full_weight)
            * condition_metrics["blurred"]["baseline_accuracy"]
        )
        aligned_routed = (
            aligned_full_weight * condition_metrics["full"]["routed_accuracy"]
            + (1.0 - aligned_full_weight)
            * condition_metrics["blurred"]["routed_accuracy"]
        )
        for prevalence in mixture_prevalences:
            baseline_accuracy = (
                (1.0 - prevalence) * aligned_baseline
                + prevalence * condition_metrics["shuffled"]["baseline_accuracy"]
            )
            routed_accuracy = (
                (1.0 - prevalence) * aligned_routed
                + prevalence * condition_metrics["shuffled"]["routed_accuracy"]
            )
            rows.append(
                {
                    "model": model_name,
                    "policy": policy_name,
                    "threshold": threshold,
                    "condition": "deployment_mixture",
                    "mixture_mismatch_prevalence": prevalence,
                    "aligned_full_weight": aligned_full_weight,
                    "n": "weighted",
                    "baseline_accuracy": baseline_accuracy,
                    "routed_accuracy": routed_accuracy,
                    "accuracy_gain": routed_accuracy - baseline_accuracy,
                }
            )
        for index, sample in enumerate(samples):
            predictions.append(
                {
                    "question_id": sample["question_id"],
                    "condition": sample["condition"],
                    "model": model_name,
                    "policy": policy_name,
                    "score": scores[index],
                    "threshold": threshold,
                    "gate": gates[index],
                    "baseline_correct": sample["baseline_correct"],
                    "fallback_correct": sample["fallback_correct"],
                    "final_correct": (
                        sample["fallback_correct"]
                        if gates[index]
                        else sample["baseline_correct"]
                    ),
                }
            )
    return rows, predictions


def bootstrap_delta_auc(
    candidate_scores: Sequence[float],
    baseline_scores: Sequence[float],
    samples: Sequence[Mapping[str, Any]],
    iterations: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if iterations <= 0:
        return None, None
    by_group: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        by_group.setdefault(str(sample["question_id"]), []).append(index)
    groups = sorted(by_group)
    labels = [bool(sample["mismatch"]) for sample in samples]
    rng = random.Random(seed)
    values: list[float] = []
    for _iteration in range(iterations):
        sampled_groups = [rng.choice(groups) for _ in groups]
        indices = [index for group in sampled_groups for index in by_group[group]]
        sampled_labels = [labels[index] for index in indices]
        candidate_auc = binary_auc(
            [candidate_scores[index] for index in indices], sampled_labels
        )
        baseline_auc = binary_auc(
            [baseline_scores[index] for index in indices], sampled_labels
        )
        if candidate_auc is not None and baseline_auc is not None:
            values.append(candidate_auc - baseline_auc)
    if not values:
        return None, None
    values.sort()
    lower = values[max(0, int(0.025 * len(values)) - 1)]
    upper = values[min(len(values) - 1, int(0.975 * len(values)))]
    return lower, upper


def run_calibration(args: argparse.Namespace) -> None:
    export_dir = args.export_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, files = load_export(export_dir)
    selected_groups, discovery = discover_heads(
        files,
        manifest,
        output_dir,
        args.top_k,
        args.max_heads_per_layer,
        args.random_head_sets,
        args.seed,
    )
    samples = build_samples(files, selected_groups)
    sets = feature_sets(samples)
    labels = [bool(sample["mismatch"]) for sample in samples]
    groups = [str(sample["question_id"]) for sample in samples]
    cv_scores: dict[str, list[float]] = {}
    models: dict[str, Any] = {}
    thresholds: dict[str, dict[str, float]] = {}
    for model_name, features in sets.items():
        cv_scores[model_name] = cross_validated_scores(
            features, labels, groups, args.folds, args.seed, args.l2
        )
        models[model_name] = fit_logistic(features, labels, args.l2)
        full_scores = logistic_scores(models[model_name], features)
        thresholds[model_name] = {
            "primary_fpr": threshold_at_fpr(
                full_scores, labels, args.target_fpr
            ),
            "fpr_0.01": threshold_at_fpr(full_scores, labels, 0.01),
            "fpr_0.05": threshold_at_fpr(full_scores, labels, 0.05),
            "fpr_0.10": threshold_at_fpr(full_scores, labels, 0.10),
            "youden": youden_threshold(full_scores, labels),
        }
    direct_scores = {
        "low_margin": [float(sample["uncertainty"][0]) for sample in samples],
        "entropy_raw": [float(sample["uncertainty"][1]) for sample in samples],
    }
    for signal_name, scores in direct_scores.items():
        cv_scores[signal_name] = scores
        thresholds[signal_name] = {
            "primary_fpr": threshold_at_fpr(scores, labels, args.target_fpr),
            "fpr_0.01": threshold_at_fpr(scores, labels, 0.01),
            "fpr_0.05": threshold_at_fpr(scores, labels, 0.05),
            "fpr_0.10": threshold_at_fpr(scores, labels, 0.10),
            "youden": youden_threshold(scores, labels),
        }
    baseline_auc = binary_auc(cv_scores["uncertainty"], labels)
    metric_rows = [
        {
            **model_metrics(
                model_name,
                scores,
                samples,
                baseline_auc,
            ),
            "evaluation": "group_cross_validation",
            "folds": args.folds,
        }
        for model_name, scores in cv_scores.items()
    ]
    write_csv(output_dir / "detector_cv_metrics.csv", metric_rows)

    metric_lookup = {str(row["model"]): row for row in metric_rows}
    primary_comparison_name = "uncertainty_plus_activation_mismatch"
    random_comparisons = [
        row
        for row in metric_rows
        if str(row["model"]).startswith("uncertainty_plus_activation_random_")
    ]
    random_null_summary: dict[str, Any] = {
        "primary_model": primary_comparison_name,
        "n_random_head_sets": len(random_comparisons),
    }
    if primary_comparison_name in metric_lookup and random_comparisons:
        observed = metric_lookup[primary_comparison_name][
            "delta_auroc_over_uncertainty"
        ]
        null_values = [
            float(row["delta_auroc_over_uncertainty"])
            for row in random_comparisons
            if row["delta_auroc_over_uncertainty"] is not None
        ]
        random_null_summary.update(
            {
                "observed_delta_auroc": observed,
                "random_delta_aurocs": null_values,
                "empirical_tie_aware_p": (
                    (
                        1
                        + sum(
                            value >= float(observed)
                            for value in null_values
                        )
                    )
                    / (1 + len(null_values))
                    if observed is not None and null_values
                    else None
                ),
                "observed_percentile_among_random": (
                    sum(value < float(observed) for value in null_values)
                    / len(null_values)
                    if observed is not None and null_values
                    else None
                ),
            }
        )
    (output_dir / "random_head_null_cv.json").write_text(
        json.dumps(jsonable(random_null_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    primary_model = (
        "uncertainty_plus_activation_mismatch"
        if "uncertainty_plus_activation_mismatch" in models
        else max(
            models,
            key=lambda name: binary_auc(cv_scores[name], labels) or float("-inf"),
        )
    )
    calibration = {
        "version": 2,
        "source_export": str(export_dir),
        "n_questions": len(files),
        "selected_head_groups": selected_groups,
        "discovery_summary": discovery,
        "models": models,
        "direct_signals": ["low_margin", "entropy_raw"],
        "thresholds": thresholds,
        "primary_model": primary_model,
        "primary_threshold_policy": "primary_fpr",
        "target_fpr": args.target_fpr,
        "folds": args.folds,
        "seed": args.seed,
        "l2": args.l2,
        "selection_uses_correctness_labels": False,
        "probe_target": "Shuffled vs {Full, Blurred}",
    }
    calibration_path = output_dir / "mismatch_calibration.json"
    calibration_path.write_text(
        json.dumps(jsonable(calibration), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    routing: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    routing_scores = {
        **{
            model_name: logistic_scores(models[model_name], sets[model_name])
            for model_name in models
        },
        **direct_scores,
    }
    for model_name in sorted(routing_scores):
        if model_name not in {"low_margin", "entropy_raw", "uncertainty"} and not model_name.startswith(
            "uncertainty_plus_activation_"
        ):
            continue
        scores = routing_scores[model_name]
        policy_thresholds = {
            name: value
            for name, value in thresholds[model_name].items()
            if name in {"primary_fpr", "youden"}
        }
        rows, sample_rows = routing_rows(
            model_name,
            scores,
            policy_thresholds,
            samples,
            args.mixture_prevalences,
            args.aligned_full_weight,
        )
        routing.extend(rows)
        predictions.extend(sample_rows)
    write_csv(output_dir / "routing_train_diagnostic.csv", routing)
    write_csv(output_dir / "detector_predictions_train.csv", predictions)
    print(f"calibration={calibration_path}")


def run_evaluation(args: argparse.Namespace) -> None:
    if args.calibration is None:
        raise ValueError("--calibration is required in evaluation mode")
    export_dir = args.export_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _manifest, files = load_export(export_dir)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    selected_groups = calibration["selected_head_groups"]
    samples = build_samples(files, selected_groups)
    sets = feature_sets(samples)
    labels = [bool(sample["mismatch"]) for sample in samples]
    models = calibration["models"]
    scores_by_model = {
        name: logistic_scores(model, sets[name])
        for name, model in models.items()
        if name in sets
    }
    scores_by_model.update(
        {
            "low_margin": [
                float(sample["uncertainty"][0]) for sample in samples
            ],
            "entropy_raw": [
                float(sample["uncertainty"][1]) for sample in samples
            ],
        }
    )
    if "uncertainty" not in scores_by_model:
        raise RuntimeError("Frozen calibration has no uncertainty baseline")
    baseline_scores = scores_by_model["uncertainty"]
    baseline_auc = binary_auc(baseline_scores, labels)
    metric_rows: list[dict[str, Any]] = []
    for model_name, scores in scores_by_model.items():
        lower, upper = bootstrap_delta_auc(
            scores,
            baseline_scores,
            samples,
            args.bootstrap,
            args.seed,
        )
        metric_rows.append(
            {
                **model_metrics(model_name, scores, samples, baseline_auc),
                "evaluation": "frozen",
                "delta_auroc_group_bootstrap_ci_low": lower,
                "delta_auroc_group_bootstrap_ci_high": upper,
                "bootstrap_iterations": args.bootstrap,
            }
        )
    write_csv(output_dir / "detector_metrics_frozen.csv", metric_rows)

    metric_lookup = {str(row["model"]): row for row in metric_rows}
    primary_comparison_name = "uncertainty_plus_activation_mismatch"
    random_comparisons = [
        row
        for row in metric_rows
        if str(row["model"]).startswith("uncertainty_plus_activation_random_")
    ]
    random_null_summary: dict[str, Any] = {
        "primary_model": primary_comparison_name,
        "n_random_head_sets": len(random_comparisons),
    }
    if primary_comparison_name in metric_lookup and random_comparisons:
        observed = metric_lookup[primary_comparison_name][
            "delta_auroc_over_uncertainty"
        ]
        null_values = [
            float(row["delta_auroc_over_uncertainty"])
            for row in random_comparisons
            if row["delta_auroc_over_uncertainty"] is not None
        ]
        random_null_summary.update(
            {
                "observed_delta_auroc": observed,
                "random_delta_aurocs": null_values,
                "empirical_tie_aware_p": (
                    (1 + sum(value >= float(observed) for value in null_values))
                    / (1 + len(null_values))
                    if observed is not None and null_values
                    else None
                ),
                "observed_percentile_among_random": (
                    sum(value < float(observed) for value in null_values)
                    / len(null_values)
                    if observed is not None and null_values
                    else None
                ),
            }
        )
    (output_dir / "random_head_null_frozen.json").write_text(
        json.dumps(jsonable(random_null_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    routing: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for model_name, scores in scores_by_model.items():
        if model_name not in {"low_margin", "entropy_raw", "uncertainty"} and not model_name.startswith(
            "uncertainty_plus_activation_"
        ):
            continue
        calibrated_thresholds = calibration["thresholds"][model_name]
        policy_thresholds = {
            name: float(value)
            for name, value in calibrated_thresholds.items()
            if name in {"primary_fpr", "fpr_0.01", "fpr_0.05", "fpr_0.10", "youden"}
        }
        rows, sample_rows = routing_rows(
            model_name,
            scores,
            policy_thresholds,
            samples,
            args.mixture_prevalences,
            args.aligned_full_weight,
        )
        routing.extend(rows)
        predictions.extend(sample_rows)
    write_csv(output_dir / "routing_metrics_frozen.csv", routing)
    write_csv(output_dir / "detector_predictions_frozen.csv", predictions)
    evaluation_summary = {
        "source_export": str(export_dir),
        "calibration": str(args.calibration.resolve()),
        "n_questions": len(files),
        "primary_model": calibration["primary_model"],
        "primary_threshold_policy": calibration["primary_threshold_policy"],
        "selection_and_threshold_frozen": True,
        "public_benchmark_transfer": "deferred",
    }
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(evaluation_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_float_list(value: str) -> list[float]:
    result = [float(part) for part in value.split(",") if part.strip()]
    if not result or any(not 0.0 <= item <= 1.0 for item in result):
        raise argparse.ArgumentTypeError("Expected comma-separated values in [0,1]")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("calibration", "evaluation"), required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-heads-per-layer", type=int, default=2)
    parser.add_argument("--random-head-sets", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--mixture-prevalences",
        type=parse_float_list,
        default=parse_float_list("0,0.01,0.05,0.1,0.25,0.5,1"),
    )
    parser.add_argument("--aligned-full-weight", type=float, default=0.8)
    args = parser.parse_args()
    if (
        args.top_k <= 0
        or args.max_heads_per_layer <= 0
        or args.random_head_sets < 0
    ):
        raise ValueError("Head selection counts must be positive")
    if not 2 <= args.folds:
        raise ValueError("--folds must be at least 2")
    if not 0.0 <= args.target_fpr <= 1.0:
        raise ValueError("--target-fpr must be in [0,1]")
    if not 0.0 <= args.aligned_full_weight <= 1.0:
        raise ValueError("--aligned-full-weight must be in [0,1]")
    if args.mode == "calibration":
        run_calibration(args)
    else:
        run_evaluation(args)


if __name__ == "__main__":
    main()
