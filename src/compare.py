"""Stage C: score every arm against every other at the selected operating point.

Selection happens on validation data and reporting on test data. Each arm's candidate pool
comes from the registry and the setting with the best validation score against the
baseline is selected. The gate is applied symmetrically: an arm that cannot pass it falls
back to the untreated baseline and the row is flagged, so a combination never degrades
silently.

Selected cells are then compared directly on test measures rather than through their
baseline margins. Missing training or evaluation output is an error rather than a skipped
row; a diverged cell leaves the pool explicitly and is reported.
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import paths as P, registry as R, metrics as M
from stats import score, val_acc_vec, acc_ok, read_metrics

BB = os.environ.get('H2H_BB', 'r18')
assert BB in R.BACKBONES, f'invalid H2H_BB: {BB}'
MODE = os.environ.get('H2H_MODE', 'sym')
assert MODE in ('sym', 'off', 'asym'), f'H2H_MODE: sym|off|asym'
GATE_US, GATE_THEM = MODE in ('sym', 'asym'), MODE == 'sym'
FAM = {m: f for f, ms in M.FAMILY.items() for m in ms}
MINE = [x for x in os.environ.get('H2H_MINE', ','.join(R.ARMS_ACTIVE)).split(',') if x]
BUCKET = os.environ.get('H2H_KOVA', '0') == '1'
REP = os.environ.get('H2H_REP', 'feat')
assert REP in ('feat', 'hybrid', 'val'), 'H2H_REP: feat|hybrid|val'
HYBRID = {'PCA': 'aux'}
ALL = R.ARMS_ACTIVE + R.ARMS_RIVAL
for _m in MINE:
    assert _m in ALL, f'invalid H2H_MINE: {_m} (secenekler: {ALL})'
ARMS = ALL
OTHERS = ['base'] + [a for a in ALL if a not in MINE]


def _nval(ds):
    import evaluate as A
    return len(A.metric_idx(ds, 'val', R.N_VAL[ds]))


def _ntest(ds):
    import evaluate as A
    return len(A.metric_idx(ds, 'test', R.N_TEST[ds]))


def to_buckets(piv):
    """Reduce the (dr, metric) pivot to two buckets: PCA as is, the three neighborhood methods
    averaged, since they carry largely the same information.
    """
    if piv is None or not BUCKET:
        return piv
    drs = list(piv.index.get_level_values(0).unique())
    komsu = [d for d in drs if d != 'PCA']
    parcalar = []
    if 'PCA' in drs:
        parcalar.append(piv.loc[['PCA']])
    if komsu:
        k = piv.loc[komsu].groupby(level='metric').mean()
        k.index = pd.MultiIndex.from_product([['komsuluk'], k.index], names=['dr', 'metric'])
        parcalar.append(k)
    return pd.concat(parcalar)


def rep_map_for(ds, arm, cell, Bval):
    """Which representation each projection is fitted on for our arms."""
    if arm not in MINE or REP == 'feat':
        return None
    if REP == 'hybrid':
        return dict(HYBRID)
    nval = _nval(ds)
    Cf = read_metrics(BB, ds, cell, 'val', nval)
    Ca = read_metrics(BB, ds, cell, 'val', nval, rep_map={d: 'aux' for d in Cf.index.get_level_values(0).unique()})
    out = {}
    for dr in Cf.index.get_level_values(0).unique():
        nf = sum((x['outcome'] == 'W') - (x['outcome'] == 'L')
                 for x in score(Bval.loc[[dr]], Cf.loc[[dr]]))
        na = sum((x['outcome'] == 'W') - (x['outcome'] == 'L')
                 for x in score(Bval.loc[[dr]], Ca.loc[[dr]]))
        out[dr] = 'aux' if na > nf else 'feat'
    return out


def select(ds, arm, Bval, base_v, gated=True):
    """Select a cell on the validation score and return it together with that score. The
    representation map is fixed on validation data and reused unchanged on test data.
    """
    nval = _nval(ds)
    bc, bn, br = None, None, None
    for c in R.arm_pool(BB, arm):
        if gated and not acc_ok(BB, ds, c, base_v):
            continue
        rm = rep_map_for(ds, arm, c, Bval)
        Cv = to_buckets(read_metrics(BB, ds, c, 'val', nval, rep_map=rm))
        if Cv is None:
            print(f'   [diverged] {BB}/{ds}/{c} — dropped from the pool', flush=True)
            continue
        o = score(Bval, Cv)
        n = sum(1 for x in o if x['outcome'] == 'W') - sum(1 for x in o if x['outcome'] == 'L')
        if bn is None or n > bn:
            bc, bn, br = c, n, rm
    return bc, bn, br


def main():
    sel = {}
    for ds in R.DATASETS:
        base_v = val_acc_vec(BB, ds, R.BASE)
        if not np.isfinite(base_v).all():
            raise RuntimeError(f'BASELINE MISSING OR DIVERGED: {BB}/{ds} — every seed is required (protocol: 10)')
        Bval = to_buckets(read_metrics(BB, ds, R.BASE, 'val', _nval(ds)))
        for arm in ARMS:
            g = GATE_US if arm in MINE else GATE_THEM
            sel[(ds, arm)] = select(ds, arm, Bval, base_v, gated=g)
        sel[(ds, 'base')] = (R.BASE, 0, None)

    print(f'\n=== SELECTIONS ({BB}; by validation score, mode={MODE} '
          f'[ours {"gated" if GATE_US else "free"}, others {"gated" if GATE_THEM else "free"}]) ===')
    for ds in R.DATASETS:
        print(f'\n{ds}')
        for arm in ARMS + ['base']:
            c, n, _ = sel[(ds, arm)]
            tag = 'GATE FAILED -> base' if c is None else c
            print(f'   {arm:12s} {tag:24s} val-net {"--" if n is None else f"{n:+d}"}')

    rows = []
    for ds in R.DATASETS:
        for tn in (None,):
            n = _ntest(ds)
            piv = {}
            def T(cell, rm=None):
                k = (cell, tuple(sorted(rm.items())) if rm else None)
                if k not in piv:
                    piv[k] = to_buckets(read_metrics(BB, ds, cell, 'test', n, rep_map=rm))
                return piv[k]
            for mine in MINE:
                mc, _, mrm = sel[(ds, mine)]
                mine_fb = mc is None
                if mine_fb: mc, mrm = R.BASE, None
                for vs in OTHERS:
                    tc, _, _ = sel[(ds, vs)]
                    fb = tc is None
                    if fb: tc = R.BASE
                    o = score(T(tc), T(mc, mrm))
                    W = sum(1 for x in o if x['outcome'] == 'W')
                    L = sum(1 for x in o if x['outcome'] == 'L')
                    rows.append(dict(dataset=ds, n=n, mine=mine, mine_cell=mc, vs=vs,
                                     vs_cell=tc, net=W - L, W=W, L=L,
                                     fb=int(fb), mine_fb=int(mine_fb),
                                     rep=';'.join(f'{k}:{v}' for k, v in sorted((mrm or {}).items()))))
                    for x in o:
                        rows.append(dict(dataset=ds, n=n, mine=mine, vs=vs,
                                         fam=FAM[x['metric']], dr=x['dr'], metric=x['metric'],
                                         s=(x['outcome'] == 'W') - (x['outcome'] == 'L'),
                                         detail=1))
    D = pd.DataFrame(rows)
    _sfx = ('' if MODE == 'sym' else f'_{MODE}') + ('' if REP == 'feat' else f'_rep-{REP}') + ('_kova' if BUCKET else '')
    if MINE != R.ARMS_ACTIVE: _sfx += '_mine-' + '-'.join(MINE)
    out = os.path.join(os.path.dirname(HERE), 'analysis', f'h2h_{BB}{_sfx}.csv')
    D.to_csv(out, index=False)

    S = D[D.get('detail').isna()] if 'detail' in D else D
    for tn in (None,):
        print(f'\n=== {BB} | mod={MODE} | fit on validation | arm vs arm (+ = ours better) ===')
        g = S
        for vs in OTHERS:
            gg = g[g.vs == vs]
            t = gg.pivot_table(index='mine', columns='dataset', values='net', aggfunc='sum')
            t = t.reindex(MINE)[[d for d in R.DATASETS if d in t.columns]]
            t['TOTAL'] = t.sum(1)
            print(f'\n--- vs {vs} ---')
            print(t.to_string())
        fbg = g[(g.fb == 1) | (g.mine_fb == 1)]
        if len(fbg):
            print('\nfallbacks to the baseline (fb/mine_fb):')
            for _, r in fbg.drop_duplicates(['dataset', 'mine', 'vs']).iterrows():
                who = 'OURS' if r.mine_fb else r.vs
                print(f'   {r.dataset}/{r.mine} vs {r.vs}: {who} failed the gate -> base')
    print(f'\nwrote: {out}\ndone', flush=True)


if __name__ == '__main__':
    main()
