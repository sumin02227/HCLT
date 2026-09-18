#!/usr/bin/env python3
"""제출 jsonl 에서 '어떤 검색 이미지를 몇 번으로 붙였는지'만 뽑는다.

submit_test.py 가 attached_ranks / attached_files 를 직접 기록하므로 검색
manifest 와 조인할 필요가 없다.  --attached-only 를 주면 실제로 붙은 문항만
남긴다.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any


def usable_candidates(
    manifest: Path | None, links: Path | None, split: str
) -> dict[str, list[dict[str, Any]]]:
    """구버전 jsonl 복원용: question_id -> 실제 파일이 있는 후보 목록(rank 순)."""
    index: dict[str, list[dict[str, Any]]] = {}
    if manifest:
        with io.open(manifest, encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("split") != split:
                    continue
                if row.get("download_status") not in {"ok", "cached"}:
                    continue
                index.setdefault(str(row["question_id"]), []).append(
                    {"rank": int(row["rank"]), "file": row.get("image_name", "")}
                )
    elif links:
        payload = json.loads(links.read_text(encoding="utf-8")).get(split) or {}
        for question_id, candidates in payload.items():
            for rank, candidate in enumerate(candidates, start=1):
                if not str(candidate.get("이미지링크") or "").strip():
                    continue
                index.setdefault(str(question_id), []).append(
                    {"rank": rank, "file": f"{split}_{question_id}_{rank}"}
                )
    for values in index.values():
        values.sort(key=lambda c: c["rank"])
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--attached-only", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="구버전 jsonl 복원용 retrieved_manifest.csv")
    parser.add_argument("--links", type=Path, default=None,
                        help="구버전 jsonl 복원용 allscan_top3_links.json (근사)")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in io.open(args.jsonl, encoding="utf-8")
        if line.strip()
    ]

    has_fields = any("attached_files" in record for record in records)
    fallback = {} if has_fields else usable_candidates(args.manifest, args.links, args.split)
    if not has_fields:
        if not fallback:
            raise SystemExit(
                "이 jsonl 에는 attached_files 가 없습니다. "
                "--manifest 또는 --links 로 복원 자료를 지정하세요."
            )
        print(
            "구버전 jsonl: attached_files 가 없어 후보 목록으로 복원합니다"
            f" ({'manifest' if args.manifest else 'links(근사)'})."
        )

    rows: list[dict[str, Any]] = []
    for record in records:
        if has_fields:
            ranks = record.get("attached_ranks") or []
            files = record.get("attached_files") or []
        else:
            # 구버전은 decision=='attached' 일 때 rank 순 첫 후보 한 장만 붙였다.
            ranks, files = [], []
            if str(record.get("retrieval_decision") or "") == "attached":
                usable = fallback.get(str(record.get("question_id", "")), [])
                if usable:
                    ranks = [usable[0]["rank"]]
                    files = [usable[0]["file"]]
        if args.attached_only and not files:
            continue
        rows.append(
            {
                "question_id": str(record.get("question_id", "")),
                "attached_ranks": ranks,
                "attached_files": files,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    attached = sum(1 for r in rows if r["attached_files"])
    total_images = sum(len(r["attached_files"]) for r in rows)
    print(f"JSON: {args.output}  ({len(rows)}문항, 부착 {attached}문항 / 이미지 {total_images}장)")

    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_output.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["question_id", "rank", "image_file"])
            for row in rows:
                if not row["attached_files"]:
                    writer.writerow([row["question_id"], "", ""])
                    continue
                # 한 문항에 여러 장이면 한 줄씩 편다.
                for rank, name in zip(row["attached_ranks"], row["attached_files"]):
                    writer.writerow([row["question_id"], rank, name])
        print(f"CSV : {args.csv_output}")


if __name__ == "__main__":
    main()
