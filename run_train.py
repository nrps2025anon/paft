#!/usr/bin/env python
"""Walk the grid and train one cell at a time.

Each cell runs in its own process, so its random number stream and its memory footprint are
independent of every other cell.

    python run_train.py
    python run_train.py --backbones r50 --datasets cub200 --seeds 0
"""
import argparse, itertools, os, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
BACKBONES = ["r18", "r50", "vit"]
DATASETS = ["svhn", "cifar100", "stl10", "cub200", "tinyimagenet"]
CELLS = ["base", "kl_cka_nbr_1_d64"]
SEEDS = list(range(10))
ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
       "NUMBA_THREADING_LAYER": "workqueue", "NUMBA_NUM_THREADS": "1",
       "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}


def main():
    ap = argparse.ArgumentParser(description="Train every cell of the grid, one process per cell.")
    ap.add_argument("--backbones", nargs="+", default=BACKBONES)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--cells", nargs="+", default=CELLS,
                    help="cell names as defined in src/registry.py, for example kl_cka_nbr_1_d64")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--dry-run", action="store_true", help="print the cells without training")
    a = ap.parse_args()

    env = dict(os.environ, **ENV)
    env.setdefault("PAFT_DATA", os.path.join(ROOT, "data"))
    env.setdefault("PAFT_ROOT", ROOT)
    todo = list(itertools.product(a.backbones, a.datasets, a.cells, a.seeds))
    print(f"{len(todo)} cells", flush=True)
    for bb, ds, cell, seed in todo:
        print(f"[train] {bb}/{ds}/{cell}/seed{seed}", flush=True)
        if a.dry_run:
            continue
        r = subprocess.run([sys.executable, "-u", os.path.join(ROOT, "src", "train.py"),
                            "--backbone", bb, "--dataset", ds, "--cell", cell,
                            "--seed", str(seed)], cwd=ROOT, env=env)
        if r.returncode:
            sys.exit(f"failed: {bb}/{ds}/{cell}/seed{seed}")


if __name__ == "__main__":
    main()
