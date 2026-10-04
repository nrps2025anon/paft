#!/usr/bin/env python
"""Fit the projections and compute the measures for every trained cell.

The work is embarrassingly parallel over cells, so it can be split across processes or
machines with --num-shards and --shard.

    python run_evaluate.py
    python run_evaluate.py --num-shards 8 --shard 3 --workers 8
"""
import argparse, os, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
       "NUMBA_THREADING_LAYER": "workqueue", "NUMBA_NUM_THREADS": "1"}


def main():
    ap = argparse.ArgumentParser(description="Project and measure every trained cell.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--backbone", help="restrict to one backbone")
    ap.add_argument("--dataset", help="restrict to one dataset")
    a = ap.parse_args()

    cmd = [sys.executable, "-u", os.path.join(ROOT, "src", "evaluate.py"),
           "--shard", str(a.shard), "--num-shards", str(a.num_shards),
           "--workers", str(a.workers)]
    if a.backbone:
        cmd += ["--backbone", a.backbone]
    if a.dataset:
        cmd += ["--dataset", a.dataset]
    env = dict(os.environ, **ENV)
    env.setdefault("PAFT_ROOT", ROOT)
    sys.exit(subprocess.run(cmd, cwd=ROOT, env=env).returncode)


if __name__ == "__main__":
    main()
