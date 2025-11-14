#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download ColliderML shards and split them into per-event Parquet files.

- Pulls shard URLs from the HF README (default) or generates ranges.
- Caches downloads, supports resume, and uses marker files to skip work.
- Writes one parquet row per event_id to: OUTPUT_DIR/<start_end>/<category>/<category>_event_<id>.parquet
"""

import argparse
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import requests

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

README_RAW_URL = "https://huggingface.co/datasets/OpenDataDetector/ColliderML_ttbar_pu0/raw/main/README.md"

OUTPUT_DIR = Path("/share/lustre/maxhart/data/colliderml/v0/ttbar_p0")
CACHE_DIR = OUTPUT_DIR / "_shards_cache"
MARKER_DIR = OUTPUT_DIR / "_processed_shards"

CATEGORIES = ("particles", "tracker_hits", "tracks", "calo_hits")

URL_RE = re.compile(
    r"https://actseos\.web\.cern\.ch/colliderml/v0/ttbar_p0/"
    r"(?P<cat>particles|tracker_hits|tracks|calo_hits)/"
    r"hard_scatter\.ttbar\.v1\.(?:truth\.particles|reco\.(?:tracker_hits|tracks|calo_hits))"
    r"\.events(?P<start>\d+)-(?P<end>\d+)\.parquet"
)

# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #

def shard_dirname(start: int, end: int) -> str:
    return f"{start}_{end}"

def event_filename(category: str, event_id: int) -> str:
    return f"{category}_event_{event_id}.parquet"

def target_event_path(category: str, event_id: int, start: int, end: int) -> Path:
    return OUTPUT_DIR / shard_dirname(start, end) / category / event_filename(category, event_id)

def marker_path(category: str, url: str) -> Path:
    fname = url.rsplit("/", 1)[-1]
    p = MARKER_DIR / category
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{fname}.done"

def cache_path(category: str, url: str) -> Path:
    fname = url.rsplit("/", 1)[-1]
    p = CACHE_DIR / category
    p.mkdir(parents=True, exist_ok=True)
    return p / fname

def ensure_base_dirs() -> None:
    for p in (OUTPUT_DIR, CACHE_DIR, MARKER_DIR):
        p.mkdir(parents=True, exist_ok=True)

def head_content_length(url: str) -> int:
    try:
        r = requests.head(url, allow_redirects=True, timeout=30)
        if r.ok and (cl := r.headers.get("Content-Length")):
            return int(cl)
    except Exception:
        pass
    return -1

def download_with_resume(url: str, dest: Path, max_retries: int = 4) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            resume_pos = tmp.stat().st_size if tmp.exists() else 0
            headers = {"Range": f"bytes={resume_pos}-"} if resume_pos > 0 else {}
            with requests.get(url, stream=True, headers=headers, timeout=120) as r:
                r.raise_for_status()
                mode = "ab" if resume_pos > 0 and r.status_code in (200, 206) else "wb"
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)

            expected = head_content_length(url)
            if expected > 0 and tmp.stat().st_size != expected:
                raise IOError(f"incomplete: {tmp.stat().st_size} != {expected}")

            os.replace(tmp, dest)
            return
        except Exception as e:
            logging.warning("Download attempt %d failed for %s: %s", attempt, url, e)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to download {url} after {max_retries} attempts")

def all_events_exist(category: str, start: int, end: int) -> bool:
    base = OUTPUT_DIR / shard_dirname(start, end) / category
    for ev in range(start, end + 1):
        if not (base / event_filename(category, ev)).exists():
            return False
    return True

def write_one_event_row(category: str, one_row: pa.Table, event_id: int, start: int, end: int) -> None:
    out = target_event_path(category, event_id, start, end)
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    pq.write_table(one_row, tmp)
    os.replace(tmp, out)

def split_shard_to_events(category: str, shard: Path, start: int, end: int) -> None:
    table = pq.read_table(shard)
    if "event_id" not in table.schema.names:
        raise RuntimeError(f"'event_id' column not found in {shard}")
    event_ids = table.column("event_id").to_pylist()
    for i, ev in enumerate(event_ids):
        ev = int(ev)
        if start <= ev <= end:
            write_one_event_row(category, table.slice(i, 1), ev, start, end)

# --------------------------------------------------------------------------- #
# Shard source discovery
# --------------------------------------------------------------------------- #

def urls_from_readme() -> Dict[str, List[Tuple[str, int, int]]]:
    logging.info("Fetching README to collect shard URLs...")
    r = requests.get(README_RAW_URL, timeout=60)
    r.raise_for_status()

    out: Dict[str, List[Tuple[str, int, int]]] = {c: [] for c in CATEGORIES}
    for m in URL_RE.finditer(r.text):
        cat, start, end = m.group("cat"), int(m.group("start")), int(m.group("end"))
        out[cat].append((m.group(0), start, end))
    for cat in out:
        out[cat].sort(key=lambda t: t[1])
    return out

def urls_from_generated(max_event: int = 100_000, step: int = 1_000) -> Dict[str, List[Tuple[str, int, int]]]:
    base = "https://actseos.web.cern.ch/colliderml/v0/ttbar_p0"
    patterns = {
        "particles": f"{base}/particles/hard_scatter.ttbar.v1.truth.particles.events{{start}}-{{end}}.parquet",
        "tracker_hits": f"{base}/tracker_hits/hard_scatter.ttbar.v1.reco.tracker_hits.events{{start}}-{{end}}.parquet",
        "tracks": f"{base}/tracks/hard_scatter.ttbar.v1.reco.tracks.events{{start}}-{{end}}.parquet",
        "calo_hits": f"{base}/calo_hits/hard_scatter.ttbar.v1.reco.calo_hits.events{{start}}-{{end}}.parquet",
    }
    out = {c: [] for c in CATEGORIES}
    for start in range(0, max_event, step):
        end = start + step - 1
        for cat in CATEGORIES:
            out[cat].append((patterns[cat].format(start=start, end=end), start, end))
    return out

def shard_index(urls_by_cat: Dict[str, List[Tuple[str, int, int]]], cats: Iterable[str]) -> Dict[Tuple[int, int], Dict[str, Tuple[str, int, int]]]:
    """ {(start,end): {category: (url,start,end), ...}} """
    idx: Dict[Tuple[int, int], Dict[str, Tuple[str, int, int]]] = {}
    for cat in cats:
        for url, s, e in urls_by_cat.get(cat, []):
            idx.setdefault((s, e), {})[cat] = (url, s, e)
    return idx

# --------------------------------------------------------------------------- #
# Processing
# --------------------------------------------------------------------------- #

def process_shard(category: str, url: str, start: int, end: int, keep_shard: bool = False) -> None:
    mark = marker_path(category, url)

    # Skip quickly if marker or outputs already exist
    if mark.exists():
        logging.info("[%s] SKIP (marker): %s", category, url)
        return
    if all_events_exist(category, start, end):
        logging.info("[%s] SKIP (already split): %s-%s", category, start, end)
        mark.write_text("ok\n")
        return

    # Ensure shard available
    local = cache_path(category, url)
    if local.exists():
        logging.info("[%s] Using cached shard: %s", category, local.name)
    else:
        logging.info("[%s] Downloading shard: %s", category, url)
        download_with_resume(url, local)

    # Split to per-event files
    logging.info("[%s] Splitting %s-%s -> %s/%s/", category, start, end, shard_dirname(start, end), category)
    split_shard_to_events(category, local, start, end)

    mark.write_text("ok\n")

    if not keep_shard:
        try:
            local.unlink()
        except Exception:
            pass

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download ColliderML shards and split to per-event parquet files (grouped by shard and category)."
    )
    ap.add_argument(
        "--categories",
        type=str,
        default=",".join(CATEGORIES),
        help=f"Comma-separated subset of: {', '.join(CATEGORIES)}",
    )
    ap.add_argument(
        "--keep-shards",
        action="store_true",
        help="Keep downloaded shard files in cache.",
    )
    ap.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (e.g. INFO, DEBUG).",
    )
    ap.add_argument(
        "--source",
        choices=("readme", "generate"),
        default="readme",
        help="Where to get shard URL list.",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    ensure_base_dirs()

    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    invalid = sorted(set(cats) - set(CATEGORIES))
    if invalid:
        logging.error("Unknown categories: %s", ", ".join(invalid))
        sys.exit(2)

    urls_by_cat = urls_from_readme() if args.source == "readme" else urls_from_generated()

    idx = shard_index(urls_by_cat, cats)
    for (start, end) in sorted(idx.keys(), key=lambda x: x[0]):
        logging.info("== Shard %s-%s ==", start, end)
        for cat in sorted(cats):
            meta = idx[(start, end)].get(cat)
            if not meta:
                logging.warning("[%s] No URL for %s-%s; skipping", cat, start, end)
                continue
            url, s, e = meta
            try:
                process_shard(cat, url, s, e, keep_shard=args.keep_shards)
            except Exception as ex:
                logging.error("Failed shard for %s %s-%s: %s", cat, s, e, ex)

if __name__ == "__main__":
    main()
