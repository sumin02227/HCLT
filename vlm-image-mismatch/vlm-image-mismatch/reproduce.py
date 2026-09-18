#!/usr/bin/env python3
"""Paper experiment entry points. --dry-run prints commands without loading models."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = {
    'qwen3-32b': ('Qwen/Qwen3-VL-32B-Instruct', 'qwen3_vl', 'qwen3_vl_32b'),
    'qwen3-8b': ('Qwen/Qwen3-VL-8B-Instruct', 'qwen3_vl', 'qwen3_vl_8b'),
    'qwen25-32b': ('Qwen/Qwen2.5-VL-32B-Instruct', 'qwen2_5_vl', 'qwen2_5_vl_32b'),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('task', choices=['mismatch', 'direct-query', 'controls',
                                  'retrieval-download', 'retrieval-fit', 'retrieval-test'])
    p.add_argument('--model', choices=MODELS, default='qwen3-32b')
    p.add_argument('--data-root', type=Path, default=ROOT / 'data')
    p.add_argument('--output-root', type=Path, default=ROOT / 'outputs')
    p.add_argument('--profile', choices=['quick', 'full'], default='full')
    p.add_argument('--split', choices=['train', 'validation'], default='validation')
    p.add_argument('--links', type=Path, help='Existing retrieval dump; downloading does not perform search')
    p.add_argument('--mode', choices=['baseline', 'naive', 'gated'], default='gated')
    p.add_argument('--max-attach', type=int, choices=[1, 2, 3], default=3)
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def input_file(data: Path, split: str) -> Path:
    manifest = data / f'ablation_manifest_{split}_portable.csv'
    return manifest if manifest.is_file() else data / f'{split}.json'


def build_commands(a):
    data, out = a.data_root.resolve(), a.output_root.resolve()
    model_id, family, label = MODELS[a.model]
    model = ['--model-id', model_id, '--model-family', family]
    train = out / f'{label}_mismatch_train_{a.profile}'
    valid = out / f'{label}_mismatch_validation_{a.profile}'
    retrieved = data / 'retrieved_images'
    manifest = data / 'retrieved_manifest.csv'
    ret = out / f'{label}_retrieval_validation'

    def command(script, *parts):
        return [sys.executable, str(ROOT / script), *map(str, parts)]

    if a.task == 'mismatch':
        return [command('run_mismatch_pipeline.py', '--dataset-root', data,
                        '--output-root', out, '--profile', a.profile,
                        '--run-label', label, '--shuffle-seed', 17, *model)]
    if a.task == 'direct-query':
        args = [*model, '--input-data', input_file(data, a.split),
                '--image-root', data / a.split, '--variants', 'V2', '--shuffle-seed', 17,
                '--canonical-size', '672,672', '--blur-radius', 12,
                '--output-dir', out / f'{label}_direct_{a.split}_{a.profile}',
                '--output-name', 'relevance_v2']
        if a.profile == 'quick':
            args += ['--limit', 30]
        return [command('b0_direct_query.py', *args)]
    if a.task == 'controls':
        return [command('analyze_dimension_matched_controls.py',
                        '--train-export', train / '01_mismatch_activation_export',
                        '--validation-export', valid / '01_mismatch_activation_export',
                        '--calibration', train / 'cpu_analysis/mismatch_calibration.json',
                        '--output-dir', out / f'{label}_controls_{a.profile}',
                        '--random-head-sets', 20, '--random-single-heads', 40,
                        '--random-projections', 0, '--seed', 17, '--l2', 1)]
    if a.task.startswith('retrieval-') and a.model != 'qwen3-32b':
        raise ValueError('The paper retrieval setting uses qwen3-32b only.')
    if a.task.startswith('retrieval-') and a.profile != 'full':
        raise ValueError('Use full for retrieval so partial caches cannot be confused with paper runs.')
    if a.task == 'retrieval-download':
        if a.links is None:
            raise ValueError('--links is required for retrieval-download')
        return [command('fetch_retrieved_images.py', '--links', a.links.resolve(),
                        '--image-root', retrieved, '--manifest', manifest,
                        '--splits', 'validation,test', '--workers', 8)]
    if a.task == 'retrieval-fit':
        return [
            command('retrieval_eval.py', *model, '--input-data', data / 'validation.json',
                    '--image-root', data / 'validation', '--retrieval-manifest', manifest,
                    '--retrieval-image-root', retrieved, '--split', 'validation',
                    '--output-dir', ret, '--conditions', 'orig,orig_top1,orig_shuffled',
                    '--canonical-size', '672,672', '--shuffle-seed', 17),
            command('analyze_retrieval.py', '--eval-dir', ret,
                    '--output-dir', ret / 'cpu_analysis', '--layers', '59,60,61,62,63',
                    '--heads-per-layer', 8, '--folds', 5, '--l2', 1, '--seed', 17),
        ]
    if a.task == 'retrieval-test':
        run_name = 'baseline' if a.mode == 'baseline' else f'{a.mode}_n{a.max_attach}'
        args = [*model, '--input-data', data / 'test.json', '--image-root', data / 'test',
                '--output-dir', out / f'{label}_retrieval_test', '--output-name', run_name,
                '--split', 'test', '--canonical-size', '672,672']
        if a.mode != 'baseline':
            args += ['--retrieval-manifest', manifest, '--retrieval-image-root', retrieved,
                     '--max-attach', a.max_attach]
        if a.mode == 'naive':
            args += ['--attach-always']
        elif a.mode == 'gated':
            args += ['--retrieval-probe', ret / 'cpu_analysis/retrieval_probe.json',
                     '--threshold-policy', 'youden']
        return [command('submit_test.py', *args)]
    raise ValueError(a.task)


def main():
    a = parse_args()
    try:
        commands = build_commands(a)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for cmd in commands:
        print(shlex.join(cmd), flush=True)
    if a.dry_run:
        return
    versions = {}
    for pkg in ('torch', 'transformers', 'numpy', 'accelerate', 'pillow', 'torchvision'):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    metadata_dir = a.output_root.resolve() / 'run_metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    record = {'task': a.task, 'python': platform.python_version(),
              'versions': versions, 'commands': commands, 'status': 'started'}
    path = metadata_dir / f'{stamp}_{a.task}.json'
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    try:
        for cmd in commands:
            subprocess.run(cmd, cwd=ROOT, check=True)
    except Exception:
        record['status'] = 'failed'
        raise
    else:
        record['status'] = 'completed'
    finally:
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
