"""Stage B: project the stored representations to two dimensions and measure them.

Each projection is fitted on the validation split only; the test split is transformed with
that same fit, so the projection never sees the points it is evaluated on. For cells that
have an auxiliary branch a second pass fits the projection on the bottleneck
representation instead. Both passes are measured against the same high-dimensional
reference, the penultimate representation, so the difference between them is the cost or
the benefit of passing through the bottleneck.

Measures are computed on the full validation split and on at most 10000 test points,
subsampled with a fixed permutation; cost grows faster than linearly in the number of
points.
"""
import argparse, os, sys, time
os.environ.setdefault('NUMBA_THREADING_LAYER', 'workqueue')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_VENDOR = os.path.join(os.path.dirname(HERE), 'vendor')
if _VENDOR not in sys.path:
    sys.path.append(_VENDOR)
if 'vendor' not in sys.path:
    sys.path.append('vendor')
import paths as P, registry as R
import metrics as M
from train import code_hash


def fit_and_transform(dr, V, T, seed):
    """Fit the projection on the validation split and return (val_2d, test_2d); the test split is
    only transformed.
    """
    if dr == 'pca':
        from sklearn.decomposition import PCA
        m = PCA(2, random_state=seed).fit(V)
        return m.transform(V), m.transform(T)
    if dr == 'tsne':
        from openTSNE import TSNE
        e = TSNE(n_components=2, initialization='pca', random_state=seed,
                 verbose=False).fit(V)
        return np.asarray(e), np.asarray(e.transform(T))
    if dr == 'umap':
        import umap
        m = umap.UMAP(n_components=2, random_state=seed).fit(V)
        return m.embedding_, m.transform(T)
    if dr == 'pacmap':
        import pacmap
        m = pacmap.PaCMAP(n_components=2, random_state=seed)
        Ev = m.fit_transform(V)
        return Ev, np.asarray(m.transform(T, basis=V))
    raise ValueError(dr)


def _reps(b, ds, cell, seed, split, keys=('feat', 'labels')):
    f = P.reps(b, ds, cell, seed, split)
    if not os.path.exists(f):
        return None
    with np.load(f) as z:
        out = []
        for k in keys:
            if k not in z.files:
                return None
            out.append(z[k])
        return out


def metric_idx(ds, split, N):
    """Indices the measures are computed on: all of validation, at most 10000 test points under a
    fixed permutation.
    """
    if split == 'val':
        return np.arange(N)
    cap = R.TEST_N_CAP
    if N <= cap:
        return np.arange(N)
    return np.sort(np.random.RandomState(0).permutation(N)[:cap])


def run_unit(b, ds, cell, seed, force=False):
    """One unit end to end: a single projection fit, both splits embedded, and the measures for both."""
    import json
    if not os.path.exists(P.done_marker(b, ds, cell, seed, 'train')):
        return f'[MISSING] no training marker: {b}/{ds}/{cell}/s{seed}'
    accf = P.accuracy(b, ds, cell, seed)
    if os.path.exists(accf) and json.load(open(accf)).get('diverged'):
        return f'[DIVERGED] {b}/{ds}/{cell}/s{seed} — not evaluated, reported explicitly'

    _v = _reps(b, ds, cell, seed, 'val')
    _t = _reps(b, ds, cell, seed, 'test')
    if _v is None or _t is None:
        return f'[MISSING] no representation: {b}/{ds}/{cell}/s{seed}'
    Vf, Vy = _v; Tf, Ty = _t
    assert len(Vf) == R.N_VAL[ds], f'validation size: {len(Vf)} != {R.N_VAL[ds]} ({ds})'
    assert len(Tf) == R.N_TEST[ds], f'test size: {len(Tf)} != {R.N_TEST[ds]} ({ds})'

    nv, nt = len(Vf), len(metric_idx(ds, 'test', len(Tf)))
    prm = R.cell_params(cell)
    reps_todo = ['feat'] + (['aux'] if prm['aux'] else [])
    markers = {}
    for rep in reps_todo:
        tag = '' if rep == 'feat' else 'aux_'
        markers[(rep, 'val')] = P.done_marker(b, ds, cell, seed, f'analyze_{tag}val_n{nv:05d}')
        markers[(rep, 'test')] = P.done_marker(b, ds, cell, seed, f'analyze_{tag}test_n{nt:05d}')
    if all(os.path.exists(m) for m in markers.values()) and not force:
        return f'[skip] {b}/{ds}/{cell}/s{seed}'

    P.ensure_seed_tree(b, ds, cell, seed)
    t0 = time.time()

    for rep in reps_todo:
        if rep == 'feat':
            SV, ST = Vf, Tf
        else:
            _va = _reps(b, ds, cell, seed, 'val', ('aux',))
            _ta = _reps(b, ds, cell, seed, 'test', ('aux',))
            if _va is None or _ta is None:
                raise RuntimeError(f'NO AUX {b}/{ds}/{cell}/s{seed}: the cell has aux=True '
                                   'but the stored representation has no aux key')
            SV, ST = _va[0], _ta[0]

        emb = {}
        for dr in R.DRS:
            Zv, Zt = fit_and_transform(dr, SV, ST, seed)
            emb[('val', dr)] = np.asarray(Zv, dtype=np.float32)
            emb[('test', dr)] = np.asarray(Zt, dtype=np.float32)

        for split, (F, y) in (('val', (Vf, Vy)), ('test', (Tf, Ty))):
            ppath = P.projections(b, ds, cell, seed, split, rep)
            tmp = ppath + f'.tmp{os.getpid()}'
            with open(tmp, 'wb') as _fh:
                np.savez_compressed(_fh, **{dr: emb[(split, dr)] for dr in R.DRS})
            os.replace(tmp, ppath)

            idx = metric_idx(ds, split, len(F))
            X, yy = F[idx], y[idx]
            n = len(idx)
            rows = []
            for dr in R.DRS:
                Z = emb[(split, dr)][idx]
                vals = M.evaluate(X, Z, yy, rng_seed=seed)
                rows += [dict(backbone=b, dataset=ds, cell=cell, seed=seed, split=split,
                              rep=rep, dr=P.DR_LABEL[dr], n=n, metric=k, value=v)
                         for k, v in vals.items()]
            mpath = P.metrics(b, ds, cell, seed, split, n, rep)
            tmp = mpath + f'.tmp{os.getpid()}'
            pd.DataFrame(rows).to_csv(tmp, index=False); os.replace(tmp, mpath)
            open(markers[(rep, split)], 'w').write(
                f'split={split}\nrep={rep}\nn={n}\nprotocol=val-fit\n'
                f'metric_reference=feat\ncode_hash={code_hash("evaluate")}\n')

    return (f'[ok] {b}/{ds}/{cell}/s{seed} val={nv} test={nt} '
            f'rep={"+".join(reps_todo)} ({time.time()-t0:.0f}s)')


def _star(t):
    u, force = t
    try:
        return run_unit(*u, force=force)
    except Exception as e:
        return f'[ERROR] {u}: {type(e).__name__}: {e}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', nargs='*'); ap.add_argument('--dataset', nargs='*')
    ap.add_argument('--cell', nargs='*'); ap.add_argument('--seed', type=int, nargs='*')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()

    units = R.units(backbones=a.backbone, datasets=a.dataset, cells_=a.cell, seeds=a.seed)
    mine = [u for i, u in enumerate(units) if i % a.num_shards == a.shard]
    print(f'[evaluate] total={len(units)} shard={len(mine)} workers={a.workers} '
          f'protocol=val-fit', flush=True)
    if a.workers <= 1:
        for u in mine:
            print(run_unit(*u, force=a.force), flush=True)
    else:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(a.workers) as pool:
            for msg in pool.imap_unordered(_star, [(u, a.force) for u in mine]):
                print(msg, flush=True)
    print('[evaluate] done', flush=True)


if __name__ == '__main__':
    main()
