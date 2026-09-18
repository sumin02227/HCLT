#!/usr/bin/env python3
"""Frozen single-head and dimension-matched controls for mismatch probes.

The selected mismatch heads come only from the Train calibration artifact.
Every control probe is fitted on Train condition labels and evaluated unchanged
on Validation.  Correctness labels are not used.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from analyze_mismatch_activations import (
    average_precision,
    binary_auc,
    jsonable,
    load_export,
    load_pt,
    write_csv,
)


def extract_split(
    export_dir: Path,
    required_layers: Sequence[int],
) -> dict[str, Any]:
    manifest, files = load_export(export_dir)
    unique_layers = sorted({int(value) for value in required_layers})
    activations: list[torch.Tensor] = []
    uncertainty: list[list[float]] = []
    labels: list[bool] = []
    question_ids: list[str] = []
    conditions: list[str] = []
    head_dim: int | None = None
    num_heads = int(manifest["num_heads"])

    for path in files:
        payload = load_pt(path)
        layers = [int(value) for value in payload["layers"]]
        layer_to_position = {layer: index for index, layer in enumerate(layers)}
        missing = [layer for layer in unique_layers if layer not in layer_to_position]
        if missing:
            raise RuntimeError(
                f"{path.name} does not contain required layers {missing}"
            )
        positions = [layer_to_position[layer] for layer in unique_layers]
        question_id = str(payload["question_id"])
        for condition in ("full", "blurred", "shuffled"):
            activation = payload["activations"][condition]
            selected = activation[positions].to(dtype=torch.float16).contiguous()
            if selected.shape[1] != num_heads:
                raise RuntimeError(
                    f"Unexpected head count in {path.name}: {selected.shape[1]}"
                )
            current_head_dim = int(selected.shape[-1])
            if head_dim is None:
                head_dim = current_head_dim
            elif current_head_dim != head_dim:
                raise RuntimeError("Inconsistent head dimensions in activation cache")
            score = payload["scores"][condition]
            activations.append(selected)
            uncertainty.append(
                [
                    -float(score["top1_margin"]),
                    float(score["candidate_entropy"]),
                ]
            )
            labels.append(condition == "shuffled")
            question_ids.append(question_id)
            conditions.append(condition)

    if not activations or head_dim is None:
        raise RuntimeError(f"No samples loaded from {export_dir}")
    return {
        "activations": torch.stack(activations),
        "uncertainty": torch.tensor(uncertainty, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.float32),
        "question_ids": question_ids,
        "conditions": conditions,
        "layers": unique_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "n_questions": len(files),
    }


def fit_predict_logistic(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    l2: float,
) -> list[float]:
    x = train_features.float()
    validation = validation_features.float()
    y = train_labels.float()
    means = x.mean(dim=0)
    scales = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    z = (x - means) / scales
    validation_z = (validation - means) / scales
    weights = torch.zeros(z.shape[1], dtype=torch.float32, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weights, bias],
        lr=0.5,
        max_iter=120,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = z @ weights + bias
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss = loss + float(l2) * weights.square().sum() / len(y)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return torch.sigmoid(validation_z @ weights + bias).tolist()


def flattened_selected(
    cube: torch.Tensor,
    slot_layer_positions: Sequence[int],
    slot_heads: Sequence[int],
) -> torch.Tensor:
    values = [
        cube[:, int(layer_position), int(head), :].float()
        for layer_position, head in zip(slot_layer_positions, slot_heads)
    ]
    return torch.stack(values, dim=1).flatten(1)


def projected_features(
    cube: torch.Tensor,
    slot_layer_positions: Sequence[int],
    coefficients: torch.Tensor,
) -> torch.Tensor:
    slot_cube = torch.stack(
        [cube[:, int(position), :, :] for position in slot_layer_positions],
        dim=1,
    ).float()
    return torch.einsum("nshd,sh->nsd", slot_cube, coefficients).flatten(1)


def metric_row(
    *,
    name: str,
    control_type: str,
    repeat: int | str,
    metadata: Mapping[str, Any],
    train_features: torch.Tensor,
    validation_features: torch.Tensor,
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    l2: float,
    uncertainty_auc: float,
    add_uncertainty: bool = False,
) -> dict[str, Any]:
    if add_uncertainty:
        train_features = torch.cat(
            [train["uncertainty"], train_features.float()], dim=1
        )
        validation_features = torch.cat(
            [validation["uncertainty"], validation_features.float()], dim=1
        )
    scores = fit_predict_logistic(
        train_features,
        train["labels"],
        validation_features,
        l2,
    )
    labels = [bool(value) for value in validation["labels"].tolist()]
    auc = binary_auc(scores, labels)
    ap = average_precision(scores, labels)
    return {
        "model": name,
        "control_type": control_type,
        "repeat": repeat,
        "with_uncertainty": add_uncertainty,
        "feature_dim": int(train_features.shape[1]),
        "train_n": int(train_features.shape[0]),
        "validation_n": int(validation_features.shape[0]),
        "validation_mismatch_n": int(sum(labels)),
        "validation_auroc": auc,
        "validation_auprc": ap,
        "delta_auroc_over_uncertainty": (
            float(auc - uncertainty_auc) if auc is not None else None
        ),
        **metadata,
    }


def null_summary(
    rows: Sequence[Mapping[str, Any]],
    selected_auc: float,
    control_type: str,
) -> dict[str, Any]:
    values = sorted(
        float(row["validation_auroc"])
        for row in rows
        if row["control_type"] == control_type
        and not bool(row["with_uncertainty"])
        and row.get("validation_auroc") is not None
    )
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "minimum": values[0],
        "median": values[len(values) // 2],
        "mean": sum(values) / len(values),
        "maximum": values[-1],
        "selected_tie_aware_p": (
            1 + sum(value >= selected_auc for value in values)
        )
        / (1 + len(values)),
        "selected_percentile": sum(value < selected_auc for value in values)
        / len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-export", type=Path, required=True)
    parser.add_argument("--validation-export", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-head-sets", type=int, default=20)
    parser.add_argument("--random-single-heads", type=int, default=40)
    parser.add_argument("--random-projections", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--l2", type=float, default=1.0)
    args = parser.parse_args()
    if min(
        args.random_head_sets,
        args.random_single_heads,
        args.random_projections,
    ) < 0:
        raise ValueError("Random control counts must be non-negative")

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    selected_heads = [
        [int(layer), int(head)]
        for layer, head in calibration["selected_head_groups"]["mismatch"]
    ]
    if not selected_heads:
        raise ValueError("Calibration contains no selected mismatch heads")
    required_layers = sorted({layer for layer, _head in selected_heads})
    train = extract_split(args.train_export.resolve(), required_layers)
    validation = extract_split(args.validation_export.resolve(), required_layers)
    if train["num_heads"] != validation["num_heads"]:
        raise RuntimeError("Train and Validation num_heads differ")
    if train["head_dim"] != validation["head_dim"]:
        raise RuntimeError("Train and Validation head_dim differ")

    layer_to_position = {
        layer: index for index, layer in enumerate(train["layers"])
    }
    validation_layer_to_position = {
        layer: index for index, layer in enumerate(validation["layers"])
    }
    slot_train_positions = [layer_to_position[layer] for layer, _ in selected_heads]
    slot_validation_positions = [
        validation_layer_to_position[layer] for layer, _ in selected_heads
    ]
    selected_indices = [head for _layer, head in selected_heads]

    uncertainty_scores = fit_predict_logistic(
        train["uncertainty"],
        train["labels"],
        validation["uncertainty"],
        args.l2,
    )
    validation_labels = [
        bool(value) for value in validation["labels"].tolist()
    ]
    uncertainty_auc_value = binary_auc(uncertainty_scores, validation_labels)
    if uncertainty_auc_value is None:
        raise RuntimeError("Uncertainty AUROC is undefined")
    rows: list[dict[str, Any]] = [
        {
            "model": "uncertainty",
            "control_type": "output_baseline",
            "repeat": "",
            "with_uncertainty": True,
            "feature_dim": 2,
            "train_n": int(train["uncertainty"].shape[0]),
            "validation_n": int(validation["uncertainty"].shape[0]),
            "validation_mismatch_n": int(sum(validation_labels)),
            "validation_auroc": uncertainty_auc_value,
            "validation_auprc": average_precision(
                uncertainty_scores, validation_labels
            ),
            "delta_auroc_over_uncertainty": 0.0,
        }
    ]

    selected_train = flattened_selected(
        train["activations"], slot_train_positions, selected_indices
    )
    selected_validation = flattened_selected(
        validation["activations"],
        slot_validation_positions,
        selected_indices,
    )
    for add_uncertainty in (False, True):
        rows.append(
            metric_row(
                name=(
                    "selected_4head_plus_uncertainty"
                    if add_uncertainty
                    else "selected_4head"
                ),
                control_type="selected_4head",
                repeat="",
                metadata={"heads": selected_heads},
                train_features=selected_train,
                validation_features=selected_validation,
                train=train,
                validation=validation,
                l2=args.l2,
                uncertainty_auc=uncertainty_auc_value,
                add_uncertainty=add_uncertainty,
            )
        )

    for slot, ((layer, head), train_position, validation_position) in enumerate(
        zip(selected_heads, slot_train_positions, slot_validation_positions)
    ):
        rows.append(
            metric_row(
                name=f"selected_single_l{layer}_h{head}",
                control_type="selected_single_head",
                repeat=slot,
                metadata={"heads": [[layer, head]]},
                train_features=train["activations"][:, train_position, head, :],
                validation_features=validation["activations"][:, validation_position, head, :],
                train=train,
                validation=validation,
                l2=args.l2,
                uncertainty_auc=uncertainty_auc_value,
            )
        )

    num_heads = int(train["num_heads"])
    rng = random.Random(args.seed)
    selected_lookup = {(layer, head) for layer, head in selected_heads}

    for repeat in range(args.random_single_heads):
        slot = rng.randrange(len(selected_heads))
        layer = selected_heads[slot][0]
        candidates = [
            head
            for head in range(num_heads)
            if (layer, head) not in selected_lookup
        ] or list(range(num_heads))
        head = rng.choice(candidates)
        train_position = slot_train_positions[slot]
        validation_position = slot_validation_positions[slot]
        rows.append(
            metric_row(
                name=f"random_single_{repeat + 1:03d}",
                control_type="layer_matched_random_single_head",
                repeat=repeat + 1,
                metadata={"heads": [[layer, head]], "slot": slot},
                train_features=train["activations"][:, train_position, head, :],
                validation_features=validation["activations"][:, validation_position, head, :],
                train=train,
                validation=validation,
                l2=args.l2,
                uncertainty_auc=uncertainty_auc_value,
            )
        )

    for repeat in range(args.random_head_sets):
        random_heads: list[int] = []
        random_pairs: list[list[int]] = []
        for layer, selected_head in selected_heads:
            candidates = [
                head
                for head in range(num_heads)
                if head != selected_head and (layer, head) not in selected_lookup
            ] or [head for head in range(num_heads) if head != selected_head]
            head = rng.choice(candidates)
            random_heads.append(head)
            random_pairs.append([layer, head])
        rows.append(
            metric_row(
                name=f"layer_matched_random_4head_{repeat + 1:03d}",
                control_type="layer_matched_random_4head",
                repeat=repeat + 1,
                metadata={"heads": random_pairs},
                train_features=flattened_selected(
                    train["activations"], slot_train_positions, random_heads
                ),
                validation_features=flattened_selected(
                    validation["activations"],
                    slot_validation_positions,
                    random_heads,
                ),
                train=train,
                validation=validation,
                l2=args.l2,
                uncertainty_auc=uncertainty_auc_value,
            )
        )

    generator = torch.Generator(device="cpu")
    for repeat in range(args.random_projections):
        generator.manual_seed(args.seed * 1000 + repeat + 1)
        coefficients = torch.randn(
            len(selected_heads), num_heads, generator=generator
        )
        coefficients = coefficients / coefficients.norm(dim=1, keepdim=True).clamp_min(
            1e-8
        )
        rows.append(
            metric_row(
                name=f"dimension_matched_projection_{repeat + 1:03d}",
                control_type="dimension_matched_random_projection",
                repeat=repeat + 1,
                metadata={"projection_seed": args.seed * 1000 + repeat + 1},
                train_features=projected_features(
                    train["activations"], slot_train_positions, coefficients
                ),
                validation_features=projected_features(
                    validation["activations"],
                    slot_validation_positions,
                    coefficients,
                ),
                train=train,
                validation=validation,
                l2=args.l2,
                uncertainty_auc=uncertainty_auc_value,
            )
        )

    mean_coefficients = torch.full(
        (len(selected_heads), num_heads),
        1.0 / num_heads,
        dtype=torch.float32,
    )
    rows.append(
        metric_row(
            name="dimension_matched_layer_mean",
            control_type="dimension_matched_layer_mean",
            repeat="",
            metadata={"layers": [layer for layer, _head in selected_heads]},
            train_features=projected_features(
                train["activations"], slot_train_positions, mean_coefficients
            ),
            validation_features=projected_features(
                validation["activations"],
                slot_validation_positions,
                mean_coefficients,
            ),
            train=train,
            validation=validation,
            l2=args.l2,
            uncertainty_auc=uncertainty_auc_value,
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "dimension_matched_control_metrics.csv", rows)
    selected_row = next(
        row
        for row in rows
        if row["control_type"] == "selected_4head"
        and not bool(row["with_uncertainty"])
    )
    selected_auc = float(selected_row["validation_auroc"])
    selected_single_aucs = [
        float(row["validation_auroc"])
        for row in rows
        if row["control_type"] == "selected_single_head"
    ]
    random_single = null_summary(
        rows, max(selected_single_aucs), "layer_matched_random_single_head"
    )
    summary = {
        "train_export": str(args.train_export.resolve()),
        "validation_export": str(args.validation_export.resolve()),
        "calibration": str(args.calibration.resolve()),
        "selected_heads": selected_heads,
        "selected_4head_validation_auroc": selected_auc,
        "selected_single_head_aurocs": selected_single_aucs,
        "best_selected_single_validation_auroc": max(selected_single_aucs),
        "uncertainty_validation_auroc": uncertainty_auc_value,
        "layer_matched_random_single_null": random_single,
        "layer_matched_random_4head_null": null_summary(
            rows, selected_auc, "layer_matched_random_4head"
        ),
        "dimension_matched_random_projection_null": null_summary(
            rows, selected_auc, "dimension_matched_random_projection"
        ),
        "selection_uses_correctness_labels": False,
        "evaluation_is_frozen": True,
        "notes": [
            "Random 4-head controls are matched to the selected head layers.",
            "Each random projection produces one virtual head per selected layer, preserving selected feature dimension.",
            "The layer-mean control is an attention-output control, not a residual-stream feature.",
        ],
    }
    (args.output_dir / "dimension_matched_control_summary.json").write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"completed={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
