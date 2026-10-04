"""The two statistical conventions used everywhere.

score(A, B) takes (dr, metric) x seed pivots, corrects the direction of the measures where
lower is better, runs a seed-paired t-test, applies Benjamini-Hochberg at q = 0.05 within
the comparison and requires a relative difference of at least 1%. A measure counts as a
win only if it is both significant and material. The reported score is wins minus losses
and is positive when B is better.

acc_ok is the accuracy gate. A setting is admissible when its drop in mean validation
accuracy is neither significant against the baseline's own seed noise, by a one-sided
one-sample t-test at p <= 0.05, nor at least 1% in relative terms. Note that the
elimination condition is a disjunction while the measure rule is a conjunction: the burden
of proof is on the eliminating side in both cases. The gate is symmetric and applies to
every arm, including ours.
"""
import json, os, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import paths as P, registry as R
import metrics as M

LOWER = M.LOWER if hasattr(M, 'LOWER') else \
    {'normalized_stress', 'mrre_high2low', 'mrre_low2high'}


def bh(p, q=0.05):
    p = np.asarray(p, float); n = len(p)
    if n == 0: return np.zeros(0, bool)
    o = np.argsort(p); s = p[o]; thr = q * (np.arange(1, n + 1) / n)
    ok = s <= thr
    out = np.zeros(n, bool)
    if ok.any():
        k = np.max(np.where(ok)[0]); out[o[:k + 1]] = True
    return out


def score(A, B):
    """A is the reference and B the target, both as (dr, metric) x seed pivots. Positive means B is
    better.
    """
    idx = A.index.intersection(B.index)
    rec = []
    for i in idx:
        a = A.loc[i].values.astype(float); b = B.loc[i].values.astype(float)
        k = min(len(a), len(b)); d = b[:k] - a[:k]
        if i[1] in LOWER: d = -d
        dm = float(np.nanmean(d))
        if np.isnan(dm): continue
        sd = np.nanstd(d, ddof=1)
        if sd < 1e-12:
            p = 0.0 if abs(dm) > 1e-12 else 1.0
        else:
            p = stats.ttest_rel(b[:k], a[:k], nan_policy='omit').pvalue
            if not np.isfinite(p): p = 1.0
        rel = abs(dm) / (abs((np.nanmean(a[:k]) + np.nanmean(b[:k])) / 2) + 1e-12)
        rec.append([i[0], i[1], dm, float(p), rel])
    if not rec: return []
    sig = bh(np.array([r[3] for r in rec]))
    out = []
    for (dr, met, dm, p, rel), sg in zip(rec, sig):
        mat = bool(sg) and rel >= 0.01
        out.append(dict(dr=dr, metric=met, dmean=dm,
                        outcome='W' if (mat and dm > 0) else 'L' if (mat and dm < 0) else 'T'))
    return out


def val_acc_vec(b, d, c):
    """Best validation accuracy per seed, ordered for the paired test. A diverged seed is NaN and
    counts as a failure at the gate.
    """
    v = []
    for s in R.SEEDS:
        f = P.accuracy(b, d, c, s)
        if os.path.exists(f):
            j = json.load(open(f))
            v.append(np.nan if j.get('diverged') else j['best_val_acc'])
        else:
            v.append(np.nan)
    return np.array(v, float)


def acc_ok(b, d, c, base_v):
    """The accuracy gate; see the module docstring for the rule."""
    v = val_acc_vec(b, d, c)
    ok = np.isfinite(v) & np.isfinite(base_v)
    if ok.sum() < len(R.SEEDS): return False
    dif = (v[ok] - base_v[ok]) * 100
    if float(np.mean(dif)) >= 0: return True
    bv = base_v[np.isfinite(base_v)]
    mb, mv = float(np.mean(bv)), float(np.mean(v[np.isfinite(v)]))
    rel = (mb - mv) / (abs(mb) + 1e-12)
    if rel >= 0.01: return False
    if len(bv) < 2: return False
    sdb = float(np.std(bv, ddof=1))
    if sdb < 1e-12:
        return mv >= mb
    tt = stats.ttest_1samp(bv, mv)
    pv = float(tt.pvalue) / 2 if float(np.mean(bv)) > mv else 1.0
    return not (np.isfinite(pv) and pv <= 0.05)


def _read_rep(b, ds, c, split, n, rep):
    """The (dr, metric) x seed pivot for one representation, or None when the files are absent."""
    import pandas as pd
    rows = []
    for s in R.SEEDS:
        f = P.metrics(b, ds, c, s, split, n, rep=rep)
        if not os.path.exists(f):
            return None
        rows.append(pd.read_csv(f)[['dr', 'metric', 'seed', 'value']])
    piv = pd.concat(rows).pivot_table(index=['dr', 'metric'], columns='seed', values='value')
    return piv if piv.shape[1] == len(R.SEEDS) else None


def read_metrics(b, ds, c, split, n, rep_map=None):
    """The (dr, metric) x seed pivot. Every expected file is opened; a missing seed is an error,
    and a diverged cell returns None and is reported.
    """
    import pandas as pd
    rows, missing = [], []
    for s in R.SEEDS:
        accf = P.accuracy(b, ds, c, s)
        if os.path.exists(accf) and json.load(open(accf)).get('diverged'):
            return None
        f = P.metrics(b, ds, c, s, split, n)
        if not os.path.exists(f):
            missing.append(f); continue
        rows.append(pd.read_csv(f)[['dr', 'metric', 'seed', 'value']])
    if missing:
        raise RuntimeError(f'MISSING METRICS {b}/{ds}/{c} {split} n={n}: {len(missing)} files '
                           f'(first: {missing[0]}) - run the evaluation stage first; every seed is required (protocol: 10)')
    d = pd.concat(rows)
    piv = d.pivot_table(index=['dr', 'metric'], columns='seed', values='value')
    if piv.shape[1] < len(R.SEEDS):
        raise RuntimeError(f'MISSING SEED {b}/{ds}/{c} {split}: {piv.shape[1]}/{len(R.SEEDS)}')
    if rep_map:
        for dr, rep in rep_map.items():
            if rep == 'feat' or dr not in piv.index.get_level_values(0):
                continue
            alt = _read_rep(b, ds, c, split, n, 'aux')
            if alt is None or dr not in alt.index.get_level_values(0):
                continue
            piv.loc[dr] = alt.loc[dr].values
    return piv
