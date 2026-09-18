#!/usr/bin/env python3
"""Run Train calibration and frozen Validation for mismatch routing."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def resolved(path: str, base: Path) -> Path:
    value = Path(path).expanduser()
    return (base / value).resolve() if not value.is_absolute() else value.resolve()


def run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def model_label(model_id: str) -> str:
    name = model_id.rsplit("/", 1)[-1].lower()
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return name or "vlm"


def runner_environment(
    *,
    mode: str,
    input_data: Path,
    image_root: Path,
    blurred_root: Path,
    output_dir: Path,
    calibration_path: Path | None,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "MISMATCH_MODE": mode,
            "MISMATCH_INPUT_DATA": str(input_data),
            "MISMATCH_IMAGE_ROOT": str(image_root),
            "MISMATCH_BLURRED_ROOT": str(blurred_root),
            "MISMATCH_OUTPUT_DIR": str(output_dir),
            "MISMATCH_CALIBRATION_PATH": (
                str(calibration_path) if calibration_path is not None else ""
            ),
        }
    )
    return environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("quick", "full"), default="full")
    parser.add_argument("--dataset-root", default="../dataset")
    parser.add_argument("--output-root", default="../outputs")
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-VL-32B-Instruct",
    )
    parser.add_argument(
        "--model-family",
        choices=("auto", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe"),
        default="auto",
    )
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--shuffle-seed", type=int, default=17)
    parser.add_argument("--force-synthetic-shuffled", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    dataset_root = resolved(args.dataset_root, script_dir)
    output_root = resolved(args.output_root, script_dir)
    train_input = dataset_root / "ablation_manifest_train_portable.csv"
    if not train_input.is_file():
        train_input = dataset_root / "train.json"
    validation_input = dataset_root / "ablation_manifest_validation_portable.csv"
    if not validation_input.is_file():
        validation_input = dataset_root / "validation.json"
    if not train_input.is_file():
        raise FileNotFoundError(f"Missing Train manifest: {train_input}")
    if not validation_input.is_file():
        raise FileNotFoundError(f"Missing Validation input: {validation_input}")

    run_label = args.run_label or (
        "qwen3_vl_32b"
        if args.model_id == "Qwen/Qwen3-VL-32B-Instruct"
        else model_label(args.model_id)
    )
    train_output = output_root / f"{run_label}_mismatch_train_{args.profile}"
    validation_output = (
        output_root / f"{run_label}_mismatch_validation_{args.profile}"
    )
    train_export = train_output / "01_mismatch_activation_export"
    validation_export = validation_output / "01_mismatch_activation_export"
    train_analysis = train_output / "cpu_analysis"
    validation_analysis = validation_output / "cpu_analysis"
    calibration_file = train_analysis / "mismatch_calibration.json"
    bootstrap = "200" if args.profile == "quick" else "2000"

    runner = script_dir / "run_vlm_mi_experiments.py"
    analyzer = script_dir / "analyze_mismatch_activations.py"
    config = script_dir / "mismatch_routing_config.json"

    runner_overrides = [
        "--model-id",
        args.model_id,
        "--shuffle-seed",
        str(args.shuffle_seed),
    ]
    if args.model_family != "auto":
        runner_overrides.extend(["--model-family", args.model_family])
    if args.force_synthetic_shuffled:
        runner_overrides.append("--force-synthetic-shuffled")

    train_environment = runner_environment(
        mode="calibration",
        input_data=train_input,
        image_root=dataset_root / "train",
        blurred_root=dataset_root / "train_blurred",
        output_dir=train_output,
        calibration_path=None,
    )
    run(
        [
            sys.executable,
            str(runner),
            "--config",
            str(config),
            "--profile",
            args.profile,
            "--only",
            "01_mismatch_activation_export",
            *runner_overrides,
        ],
        cwd=script_dir,
        env=train_environment,
    )
    run(
        [
            sys.executable,
            str(analyzer),
            "--mode",
            "calibration",
            "--export-dir",
            str(train_export),
            "--output-dir",
            str(train_analysis),
            "--bootstrap",
            bootstrap,
        ],
        cwd=script_dir,
    )

    validation_environment = runner_environment(
        mode="evaluation",
        input_data=validation_input,
        image_root=dataset_root / "validation",
        blurred_root=dataset_root / "validation_blurred",
        output_dir=validation_output,
        calibration_path=calibration_file,
    )
    run(
        [
            sys.executable,
            str(runner),
            "--config",
            str(config),
            "--profile",
            args.profile,
            "--only",
            "01_mismatch_activation_export",
            *runner_overrides,
        ],
        cwd=script_dir,
        env=validation_environment,
    )
    run(
        [
            sys.executable,
            str(analyzer),
            "--mode",
            "evaluation",
            "--export-dir",
            str(validation_export),
            "--output-dir",
            str(validation_analysis),
            "--calibration",
            str(calibration_file),
            "--bootstrap",
            bootstrap,
        ],
        cwd=script_dir,
    )
    print(f"완료: {validation_analysis}")


if __name__ == "__main__":
    main()
