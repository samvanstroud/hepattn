#!/usr/bin/env python3
import os
import re
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import requests
import pyarrow as pa
import pyarrow.parquet as pq

README_RAW_URL = "https://huggingface.co/datasets/OpenDataDetector/ColliderML_ttbar_pu0/raw/main/README.md"

# Base output directory
OUTPUT_DIR = Path("/share/lustre/maxhart/data/colliderml/v0/ttbar_p0")
# Cache for downloaded shards
CACHE_DIR = OUTPUT_DIR / "_shards_cache"
# Markers to avoid re-processing shards
MARKER_DIR = OUTPUT_DIR / "_processed_shards"

CATEGORIES = {"particles", "tracker_hits", "tracks", "calo_hits"}

def event_filename(category: str, event_id: int) -> str:
    return f"{category}_event_{event_id}.parquet"

def shard_dirname(start: int, end: int) -> str:
    return f"{start}_{end}"

def target_event_path(category: str, event_id: int, start: int, end: int) -> Path:
    return OUTPUT_DIR / shard_dirname(start, end) / category / event_filename(category, event_id)

URL_RE = re.compile(
    r"https://actseos\.web\.cern\.ch/colliderml/v0/ttbar_p0/(?P<cat>particles|tracker_hits|tracks|calo_hits)/"
    r"hard_scatter\.ttbar\.v1\.(?:truth\.particles|reco\.(?:tracker_hits|tracks|calo_hits))\.events(?P<start>\d+)-(?P<end>\d+)\.parquet"
)

def fetch_url_list_from_readme() -> Dict[str, List[Tuple[str, int, int]]]:
    logging.info("Fetching README to collect shard URLs...")
    resp = requests.get(README_RAW_URL, timeout=60)
    resp.raise_for_status()
    urls_by_cat: Dict[str, List[Tuple[str, int, int]]] = {"particles": [], "tracker_hits": [], "tracks": [], "calo_hits": []}
    for m in URL_RE.finditer(resp.text):
        cat = m.group("cat")
        start = int(m.group("start"))
        end = int(m.group("end"))
        url = m.group(0)
        if cat in urls_by_cat:
            urls_by_cat[cat].append((url, start, end))
    for cat in urls_by_cat:
        urls_by_cat[cat].sort(key=lambda t: t[1])
    return urls_by_cat

def ensure_dirs():
    for p in [OUTPUT_DIR, CACHE_DIR, MARKER_DIR]:
        p.mkdir(parents=True, exist_ok=True)

def shard_marker_path(category: str, url: str) -> Path:
    fname = url.rsplit("/", 1)[-1]
    d = MARKER_DIR / category
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{fname}.done"

def local_shard_path(category: str, url: str) -> Path:
    fname = url.rsplit("/", 1)[-1]
    d = CACHE_DIR / category
    d.mkdir(parents=True, exist_ok=True)
    return d / fname

def head_content_length(url: str) -> int:
    try:
        r = requests.head(url, allow_redirects=True, timeout=30)
        if r.status_code == 200 and "Content-Length" in r.headers:
            return int(r.headers["Content-Length"])
    except Exception:
        pass
    return -1

def download_with_resume(url: str, dest: Path, max_retries: int = 4) -> None:
    dest_tmp = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while True:
        attempt += 1
        try:
            resume_pos = dest_tmp.stat().st_size if dest_tmp.exists() else 0
            headers = {}
            if resume_pos > 0:
                headers["Range"] = f"bytes={resume_pos}-"
            with requests.get(url, stream=True, headers=headers, timeout=120) as r:
                r.raise_for_status()
                mode = "ab" if resume_pos > 0 and r.status_code in (206, 200) else "wb"
                with open(dest_tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
            cl = head_content_length(url)
            if cl > 0 and dest_tmp.stat().st_size != cl:
                raise IOError(f"Downloaded size {dest_tmp.stat().st_size} != Content-Length {cl}")
            os.replace(dest_tmp, dest)
            return
        except Exception as e:
            logging.warning(f"Download attempt {attempt} failed for {url}: {e}")
            if attempt >= max_retries:
                raise
            time.sleep(2 ** attempt)

def all_events_exist(category: str, start: int, end: int) -> bool:
    base = OUTPUT_DIR / shard_dirname(start, end) / category
    for ev in range(start, end + 1):
        if not (base / event_filename(category, ev)).exists():
            return False
    return True

def write_one_event_row(category: str, one_row_table: pa.Table, event_id: int, start: int, end: int):
    out_path = target_event_path(category, event_id, start, end)
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    pq.write_table(one_row_table, tmp_path)
    os.replace(tmp_path, out_path)

def split_shard_to_events(category: str, shard_path: Path, start: int, end: int):
    table = pq.read_table(shard_path)
    if "event_id" not in table.schema.names:
        raise RuntimeError(f"'event_id' column not found in {shard_path}")
    event_ids = table.column("event_id").to_pylist()
    nrows = table.num_rows
    for i in range(nrows):
        ev_id = int(event_ids[i])
        if ev_id < start or ev_id > end:
            continue
        write_one_event_row(category, table.slice(i, 1), ev_id, start, end)

def process_shard(category: str, url: str, start: int, end: int, keep_shard: bool = False):
    marker = shard_marker_path(category, url)
    if marker.exists():
        logging.info(f"[{category}] SKIP (marker): {url}")
        return

    if all_events_exist(category, start, end):
        logging.info(f"[{category}] SKIP (already split): {start}-{end}")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("ok\n")
        return

    local_path = local_shard_path(category, url)
    if not local_path.exists():
        logging.info(f"[{category}] Downloading shard {url}")
        download_with_resume(url, local_path)
    else:
        logging.info(f"[{category}] Using cached shard: {local_path.name}")

    logging.info(f"[{category}] Splitting {start}-{end} from {local_path.name} -> {shard_dirname(start, end)}/{category}/")
    split_shard_to_events(category, local_path, start, end)

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok\n")

    if not keep_shard:
        try:
            local_path.unlink()
        except Exception:
            pass

def build_shard_index(urls_by_cat: Dict[str, List[Tuple[str, int, int]]], cats: List[str]) -> Dict[Tuple[int, int], Dict[str, Tuple[str, int, int]]]:
    """
    Return: {(start,end): {category: (url,start,end), ...}, ...}
    """
    idx: Dict[Tuple[int, int], Dict[str, Tuple[str, int, int]]] = {}
    for cat in cats:
        for url, start, end in urls_by_cat.get(cat, []):
            key = (start, end)
            if key not in idx:
                idx[key] = {}
            idx[key][cat] = (url, start, end)
    return idx

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download ColliderML shards and split to per-event parquet files (grouped by shard and category), processing by shard first.")
    parser.add_argument("--categories", type=str, default="particles,tracker_hits,tracks,calo_hits",
                        help="Comma-separated list of categories to process.")
    parser.add_argument("--keep-shards", action="store_true", help="Keep downloaded shard files in cache.")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--source", choices=["readme", "generate"], default="readme",
                        help="Where to get shard URL list: 'readme' scrapes the HF README; 'generate' builds ranges programmatically.")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    ensure_dirs()

    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    invalid = set(cats) - CATEGORIES
    if invalid:
        logging.error(f"Unknown categories: {sorted(invalid)}")
        sys.exit(2)

    if args.source == "readme":
        urls_by_cat = fetch_url_list_from_readme()
    else:
        base = "https://actseos.web.cern.ch/colliderml/v0/ttbar_p0"
        patterns = {
            "particles": base + "/particles/hard_scatter.ttbar.v1.truth.particles.events{start}-{end}.parquet",
            "tracker_hits": base + "/tracker_hits/hard_scatter.ttbar.v1.reco.tracker_hits.events{start}-{end}.parquet",
            "tracks": base + "/tracks/hard_scatter.ttbar.v1.reco.tracks.events{start}-{end}.parquet",
            "calo_hits": base + "/calo_hits/hard_scatter.ttbar.v1.reco.calo_hits.events{start}-{end}.parquet",
        }
        urls_by_cat = {k: [] for k in CATEGORIES}
        for start in range(0, 100000, 1000):
            end = start + 999
            for cat in CATEGORIES:
                urls_by_cat[cat].append((patterns[cat].format(start=start, end=end), start, end))

    # Build a shard-first index and iterate by shard, then by category
    shard_index = build_shard_index(urls_by_cat, cats)
    for (start, end) in sorted(shard_index.keys(), key=lambda x: x[0]):
        logging.info(f"== Shard {start}-{end} ==")
        for cat in sorted(cats):
            meta = shard_index[(start, end)].get(cat)
            if meta is None:
                logging.warning(f"[{cat}] No shard URL listed for {start}-{end}; skipping")
                continue
            url, s, e = meta
            try:
                process_shard(cat, url, s, e, keep_shard=args.keep_shards)
            except Exception as ex:
                logging.error(f"Failed shard for {cat} {s}-{e}: {ex}")

if __name__ == "__main__":
    main()
