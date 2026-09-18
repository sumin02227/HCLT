#!/usr/bin/env python3
"""Download retrieved candidate images and emit a manifest.

Input is the retrieval dump keyed by split -> question_id -> [candidate, ...],
where each candidate carries 표제어 / 설명 / 유사도 / 이미지링크 / 페이지링크.

Only candidates with a usable 이미지링크 are downloaded.  Files are named
{split}_{question_id}_{rank}.{ext} so the manifest and the image root stay in
sync, and re-running skips anything already on disk.

The manifest keeps every candidate, downloaded or not, because 표제어/설명 are
usable as text evidence even when no image exists.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
}
USER_AGENT = "Mozilla/5.0 (compatible; vlm-mi-research/1.0)"


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
        writer.writerows(rows)


def extension_for(url: str, content_type: str | None) -> str:
    if content_type:
        base = content_type.split(";")[0].strip().lower()
        if base in EXTENSIONS:
            return EXTENSIONS[base]
    suffix = Path(url.split("?")[0]).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp"} else ".jpg"


def download(
    url: str, destination_stem: Path, timeout: float, retries: int
) -> tuple[Path | None, str]:
    for existing in destination_stem.parent.glob(destination_stem.name + ".*"):
        if existing.stat().st_size > 0:
            return existing, "cached"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last = ""
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
                content_type = response.headers.get("Content-Type")
            if not payload:
                last = "empty body"
                continue
            path = destination_stem.with_suffix(extension_for(url, content_type))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            return path, "ok"
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as error:
            last = f"{type(error).__name__}: {error}"
            if attempt < retries:
                time.sleep(1.0 + attempt)
    return None, last or "failed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--links", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits", default="train,validation,test")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    payload = json.loads(args.links.read_text(encoding="utf-8"))
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    jobs: list[dict[str, Any]] = []
    for split in splits:
        questions = payload.get(split) or {}
        for question_id in sorted(questions):
            for rank, candidate in enumerate(questions[question_id], start=1):
                url = str(candidate.get("이미지링크") or "").strip()
                jobs.append(
                    {
                        "split": split,
                        "question_id": str(question_id),
                        "rank": rank,
                        "source": candidate.get("출처", ""),
                        "title": candidate.get("표제어", ""),
                        "description": candidate.get("설명", ""),
                        "similarity": candidate.get("유사도", ""),
                        "image_url": url,
                        "page_url": candidate.get("페이지링크", ""),
                    }
                )
    if args.limit:
        jobs = jobs[: args.limit]

    downloadable = [job for job in jobs if job["image_url"]]
    print(
        f"후보 {len(jobs)}개 중 이미지링크 있는 것 {len(downloadable)}개 "
        f"({len(downloadable) / max(1, len(jobs)):.1%})",
        flush=True,
    )

    def work(job: dict[str, Any]) -> dict[str, Any]:
        stem = args.image_root / f"{job['split']}_{job['question_id']}_{job['rank']}"
        path, status = download(job["image_url"], stem, args.timeout, args.retries)
        job = dict(job)
        job["image_name"] = path.name if path else ""
        job["download_status"] = status
        if path:
            job["bytes"] = path.stat().st_size
            job["sha1"] = hashlib.sha1(path.read_bytes()).hexdigest()[:16]
        return job

    results: list[dict[str, Any]] = [job for job in jobs if not job["image_url"]]
    for job in results:
        job["image_name"] = ""
        job["download_status"] = "no_link"

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, job): job for job in downloadable}
        for future in as_completed(futures):
            results.append(future.result())
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(downloadable)}", flush=True)

    results.sort(key=lambda r: (r["split"], r["question_id"], r["rank"]))
    write_csv(args.manifest, results)

    ok = sum(1 for r in results if r["download_status"] in {"ok", "cached"})
    print(f"\n저장 완료: {ok}/{len(downloadable)}  manifest={args.manifest}")
    for split in splits:
        rows = [r for r in results if r["split"] == split]
        with_image = {
            r["question_id"] for r in rows if r["download_status"] in {"ok", "cached"}
        }
        questions = {r["question_id"] for r in rows}
        print(
            f"  [{split}] 문항 {len(questions)}개 중 "
            f"이미지 확보 {len(with_image)}개 ({len(with_image)/max(1,len(questions)):.1%})"
        )
    failed = [r for r in results if r["download_status"] not in {"ok", "cached", "no_link"}]
    if failed:
        print(f"  실패 {len(failed)}건 (manifest의 download_status 확인)")


if __name__ == "__main__":
    main()
