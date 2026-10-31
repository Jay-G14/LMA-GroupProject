"""Download every distribution in a pip install report with parallel byte-range requests, verifying sha256.

On Ada a single connection to PyPI's file CDN runs at ~0.1 MB/s while 48 parallel ranges reach ~4.4 MB/s,
so `pip install` alone cannot fetch vLLM's ~3.5 GB of wheels in reasonable time. Resolve first with
`pip install --dry-run --ignore-installed --report report.json -r requirements`, then:

  python scripts/ada/fetch_wheels.py report.json /share1/$USER/wheels
  pip install --no-index --find-links /share1/$USER/wheels -r requirements
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

CHUNK = 8 * 1024 * 1024
WORKERS = int(os.environ.get("FETCH_WORKERS", "64"))


def size_of(url: str) -> int:
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Range"].split("/")[-1])


def fetch_chunk(url: str, path: Path, start: int, end: int) -> int:
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) != end - start + 1:
                raise IOError(f"short read {len(data)} of {end - start + 1}")
            with open(path, "r+b") as f:
                f.seek(start)
                f.write(data)
            return len(data)
        except Exception as e:  # retried with backoff, then raised
            if attempt == 7:
                raise RuntimeError(f"{url} bytes {start}-{end}: {e}")
            time.sleep(min(30, 2 ** attempt))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    report, out = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for inst in report["install"]:
        info = inst["download_info"]
        url, want = info["url"], info.get("archive_info", {}).get("hashes", {}).get("sha256")
        path = out / url.split("/")[-1].split("#")[0]
        if path.exists() and want and sha256(path) == want:
            continue
        items.append((url, path, want))
    print(f"{len(report['install'])} distributions; {len(items)} to download", flush=True)
    sizes = {}
    with cf.ThreadPoolExecutor(32) as ex:
        for (url, path, _), n in zip(items, ex.map(lambda it: size_of(it[0]), items)):
            sizes[url] = n
            with open(path, "wb") as f:
                f.truncate(n)
    total = sum(sizes.values())
    print(f"{total / 1e9:.2f} GB in {len(items)} files", flush=True)
    jobs = [(url, path, s, min(s + CHUNK, sizes[url]) - 1) for url, path, _ in items for s in range(0, sizes[url], CHUNK)]
    t0, done = time.time(), 0
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        for i, n in enumerate(ex.map(lambda j: fetch_chunk(*j), jobs), 1):
            done += n
            if i % 50 == 0 or i == len(jobs):
                print(f"  {done / 1e9:.2f}/{total / 1e9:.2f} GB, {done / 1e6 / (time.time() - t0):.1f} MB/s", flush=True)
    bad = [p.name for url, p, want in items if want and sha256(p) != want]
    if bad:
        sys.exit(f"sha256 mismatch: {bad}")
    print(f"all {len(items)} files verified (sha256)", flush=True)


if __name__ == "__main__":
    main()
