#!/usr/bin/env python3
"""Download the full NSW Opal Patronage dataset from the TfNSW open-data S3 bucket.

The dataset is exposed as a public S3 listing (no auth) under the prefix
``Opal_Patronage/``, organised into monthly folders of daily files named
``Opal_Patronage_YYYYMMDD.txt``. As the Data Browser page notes, requests must
include the ``Referer: https://opendata.transport.nsw.gov.au/`` header.

This mirrors the bucket into ``data/raw/<YYYY-MM>/...``. Re-running only fetches
files that are missing or zero-byte, so it's safe to use as an incremental sync.

Uses only the Python standard library (no third-party deps).

Usage:
    python scripts/download_opal_patronage.py                 # download everything
    python scripts/download_opal_patronage.py --dest data/raw # custom destination
    python scripts/download_opal_patronage.py --workers 30    # tune parallelism
    python scripts/download_opal_patronage.py --force         # re-download all
    python scripts/download_opal_patronage.py --list-only     # just print the keys
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BUCKET_URL = "https://opendata-tpa.transport.nsw.gov.au/"
PREFIX = "Opal_Patronage/"
REFERER = "https://opendata.transport.nsw.gov.au/"
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def _get(url: str, timeout: int = 60) -> bytes:
    """GET a URL with the required Referer header and return the response body."""
    req = urllib.request.Request(url, headers={"Referer": REFERER})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def list_keys() -> list[str]:
    """Return all ``.txt`` object keys under the prefix, following S3 pagination."""
    keys: list[str] = []
    marker = ""
    while True:
        query = urllib.parse.urlencode({"prefix": PREFIX, "marker": marker})
        root = ET.fromstring(_get(f"{BUCKET_URL}?{query}"))

        contents = root.findall("s3:Contents/s3:Key", S3_NS)
        keys.extend(el.text for el in contents if el.text and el.text.endswith(".txt"))

        truncated = root.findtext("s3:IsTruncated", default="false", namespaces=S3_NS)
        if truncated.lower() != "true" or not contents:
            break
        # Next page starts after the last key returned in this page.
        marker = contents[-1].text
    return keys


def download_one(key: str, dest_root: Path, force: bool) -> tuple[str, str]:
    """Download a single key. Returns (key, status) where status is ok/skip/error:...."""
    # Strip the leading "Opal_Patronage/" so we mirror only the month subfolders.
    rel = key[len(PREFIX):] if key.startswith(PREFIX) else key
    out_path = dest_root / rel
    if not force and out_path.exists() and out_path.stat().st_size > 0:
        return key, "skip"
    try:
        # Quote the path but keep "/" so the key structure is preserved.
        body = _get(BUCKET_URL + urllib.parse.quote(key))
        if not body:
            return key, "error: empty body"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(body)
        return key, "ok"
    except (urllib.error.URLError, OSError) as exc:
        return key, f"error: {exc}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dest", default="data/raw", help="destination directory (default: data/raw)")
    parser.add_argument("--workers", type=int, default=20, help="parallel downloads (default: 20)")
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    parser.add_argument("--list-only", action="store_true", help="list keys and exit without downloading")
    args = parser.parse_args(argv)

    dest_root = Path(args.dest)

    print("Enumerating bucket ...", file=sys.stderr)
    keys = list_keys()
    print(f"Found {len(keys)} files under {PREFIX}", file=sys.stderr)

    if args.list_only:
        for key in keys:
            print(key)
        return 0

    ok = skipped = 0
    errors: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_one, k, dest_root, args.force) for k in keys]
        for i, fut in enumerate(as_completed(futures), 1):
            key, status = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skipped += 1
            else:
                errors.append((key, status))
            if i % 200 == 0 or i == len(keys):
                print(f"  {i}/{len(keys)} processed", file=sys.stderr)

    print(f"\nDone: {ok} downloaded, {skipped} already present, {len(errors)} failed.", file=sys.stderr)
    if errors:
        print("Failures:", file=sys.stderr)
        for key, status in errors[:50]:
            print(f"  {key}: {status}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
