"""The inventory of the experiment: which cells exist and what each one means.

Cell names are generated from a formula rather than listed, and `protocol.json` is the
single configuration source. It also stages the scope: the full grammar of arms and doses
is always resolvable, while the active arms and doses define what is currently counted and
run, so widening the scope is a line in the configuration rather than a code change.

Name and parameters are tested round-trip: `python src/registry.py`.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

with open(os.path.join(ROOT, 'protocol.json')) as _f:
    PROTOCOL = json.load(_f)

SEEDS = list(PROTOCOL['seeds'])
BACKBONES = list(PROTOCOL['backbones'])
DATASETS = list(PROTOCOL['datasets'])
DRS = list(PROTOCOL['eval']['drs'])
TEST_N_CAP = int(PROTOCOL['eval']['metric_n_test_cap'])
DOSES = [float(x) for x in PROTOCOL['doses']]
DOSES_ACTIVE = [float(x) for x in PROTOCOL['doses_active']]
ARMS_ACTIVE = list(PROTOCOL['arms_active'])
DOSE_WAVES = [[float(x) for x in w] for w in PROTOCOL.get('dose_waves', [DOSES_ACTIVE])]
assert sorted(d for w in DOSE_WAVES for d in w) == sorted(DOSES_ACTIVE), \
    'dose_waves and doses_active do not match'


def dose_wave(dose):
    """Submission order of a dose. Scope, analysis and tables treat all waves as one campaign."""
    for i, w in enumerate(DOSE_WAVES):
        if any(abs(dose - x) < 1e-12 for x in w):
            return i
    return len(DOSE_WAVES)
D_BOTH = [int(x) for x in PROTOCOL['d_both']]
ARMS_OURS = list(PROTOCOL['arms_ours'])
ARMS_RIVAL = list(PROTOCOL['arms_rival'])
NECK_DIMS = {b: [int(x) for x in v] for b, v in PROTOCOL.get('neck_dims', {}).items()}
NECK_WAVE = int(PROTOCOL.get('neck_wave', 99))
RANK_DOSES = [float(x) for x in PROTOCOL.get('rank_doses', [])]
RANK_WAVE = int(PROTOCOL.get('rank_wave', 99))
CENTER_DOSES = [float(x) for x in PROTOCOL.get('center_doses', [])]
CENTER_WAVE = int(PROTOCOL.get('center_wave', 99))
BASE = 'base'
N_TEST = {k: int(v) for k, v in PROTOCOL['n_test'].items()}
N_VAL = {k: int(v) for k, v in PROTOCOL['n_val'].items()}

ARM_TERMS = {'kl_cka_nbr': ('kl', 'cka', 'nbr'), 'kl_cka': ('kl', 'cka'),
             'kl_nbr': ('kl', 'nbr'), 'cka_nbr': ('cka', 'nbr'),
             'kl': ('kl',), 'cka': ('cka',), 'nbr': ('nbr',)}


def dose_code(a):
    """Dose to name fragment: 0.25 becomes 025. No separator, no collisions."""
    return f'{a:g}'.replace('.', '')

DOSE2VAL = {dose_code(a): a for a in DOSES}
assert len(DOSE2VAL) == len(DOSES), 'dose encoding collides'


def rule_d(backbone):
    """The single bottleneck width used by the ablation arms."""
    return int(PROTOCOL['d_rule_ablation'][backbone])


def cell_name(arm, dose, d):
    return f'{arm}_{dose_code(dose)}_d{int(d)}'


def neck_cell_name(nd):
    return f'neck_d{int(nd)}'


def neck_dims(backbone):
    return list(NECK_DIMS.get(backbone, []))


def rank_cell_name(dose):
    return f'rank_{dose_code(dose)}'


def center_cell_name(dose):
    return f'center_{dose_code(dose)}'


def cells(backbone):
    """The active cells of one backbone in a deterministic order, independent of the dataset."""
    out = [BASE]
    for arm in ARMS_ACTIVE + ARMS_RIVAL:
        if arm == 'neck':
            out += [neck_cell_name(nd) for nd in neck_dims(backbone)]
            continue
        if arm == 'rank':
            out += [rank_cell_name(a) for a in RANK_DOSES]
            continue
        if arm == 'center':
            out += [center_cell_name(a) for a in CENTER_DOSES]
            continue
        ds_list = D_BOTH if (arm == 'kl_cka_nbr' or arm in ARMS_RIVAL) else [rule_d(backbone)]
        for a in DOSES_ACTIVE:
            for d in ds_list:
                out.append(cell_name(arm, a, d))
    return out


def arm_pool(backbone, arm):
    """The candidate pool of one arm; selection happens within it."""
    if arm == BASE:
        return [BASE]
    if arm == 'neck':
        return [neck_cell_name(nd) for nd in neck_dims(backbone)]
    if arm == 'rank':
        return [rank_cell_name(a) for a in RANK_DOSES]
    if arm == 'center':
        return [center_cell_name(a) for a in CENTER_DOSES]
    ds_list = D_BOTH if (arm == 'kl_cka_nbr' or arm in ARMS_RIVAL) else [rule_d(backbone)]
    return [cell_name(arm, a, d) for a in DOSES_ACTIVE for d in ds_list]


def cell_params(cell):
    """Cell name to training parameters. The auxiliary branch exists only for our arms, since all
    three terms are defined through it; the top-d arm penalizes the penultimate
    representation directly and uses d only as the target rank.
    """
    import paths as P
    if cell == BASE:
        return dict(arm='base', dose=0.0, w_kl=0.0, w_cka=0.0, w_nbr=0.0,
                    w_rank=0.0, w_center=0.0, rank_mode='topd', proj_dim=2, aux=False,
                    neck_dim=None)
    mc = P.CELL_CENTER_RE.match(cell)
    if mc:
        _d = {dose_code(a): a for a in CENTER_DOSES}[mc['dose']]
        return dict(arm='center', dose=_d, w_kl=0.0, w_cka=0.0, w_nbr=0.0, w_rank=0.0,
                    w_center=_d, rank_mode='topd', proj_dim=2, aux=False, neck_dim=None)
    mr = P.CELL_RANK_RE.match(cell)
    if mr:
        _d = {v: k for k, v in ((a, dose_code(a)) for a in RANK_DOSES)}[mr['dose']]
        return dict(arm='rank', dose=_d, w_kl=0.0, w_cka=0.0, w_nbr=0.0,
                    w_rank=_d, w_center=0.0, rank_mode='nuclear', proj_dim=2, aux=False,
                    neck_dim=None)
    mn = P.CELL_NECK_RE.match(cell)
    if mn:
        return dict(arm='neck', dose=0.0, w_kl=0.0, w_cka=0.0, w_nbr=0.0,
                    w_rank=0.0, w_center=0.0, rank_mode='topd', proj_dim=2, aux=False,
                    neck_dim=int(mn['nd']))
    m = P.CELL_RE.match(cell)
    assert m, f'cell name does not match the grammar: {cell}'
    arm, a, d = m['arm'], DOSE2VAL[m['dose']], int(m['d'])
    if arm == 'trank':
        return dict(arm=arm, dose=a, w_kl=0.0, w_cka=0.0, w_nbr=0.0,
                    w_rank=a, w_center=0.0, rank_mode='topd', proj_dim=d, aux=False,
                    neck_dim=None)
    t = ARM_TERMS[arm]
    return dict(arm=arm, dose=a,
                w_kl=a if 'kl' in t else 0.0,
                w_cka=a if 'cka' in t else 0.0,
                w_nbr=a if 'nbr' in t else 0.0,
                w_rank=0.0, w_center=0.0, rank_mode='topd', proj_dim=d, aux=True,
                neck_dim=None)


def metric_n(dataset, split):
    """The n the measures are computed at: the full validation split, and the test split capped
    under a fixed permutation.
    """
    return N_VAL[dataset] if split == 'val' else min(TEST_N_CAP, N_TEST[dataset])


def units(backbones=None, datasets=None, cells_=None, seeds=None):
    """The expected (backbone, dataset, cell, seed) quadruples, ordered and unique."""
    out = []
    for b in (backbones or BACKBONES):
        cs = cells_ if cells_ is not None else cells(b)
        for d in (datasets or DATASETS):
            for c in cs:
                for s in (seeds if seeds is not None else SEEDS):
                    out.append((b, d, c, int(s)))
    return out


def _selftest():
    import paths as P
    bad = []
    for b in BACKBONES:
        exp = 1 + sum((len(neck_dims(b)) if a == 'neck' else
                       len(RANK_DOSES) if a == 'rank' else
                       len(CENTER_DOSES) if a == 'center' else
                       len(DOSES_ACTIVE) *
                       (len(D_BOTH) if (a == 'kl_cka_nbr' or a in ARMS_RIVAL) else 1))
                      for a in ARMS_ACTIVE + ARMS_RIVAL)
        cs = cells(b)
        assert len(cs) == exp, f'{b}: {len(cs)} cells, expected {exp}'
        assert len(set(cs)) == len(cs), f'{b}: duplicate cell name'
        for c in cs:
            if not P.is_cell(c):
                bad.append((b, c, 'dilbilgisi')); continue
            p = cell_params(c)
            if c == BASE:
                continue
            rt = (neck_cell_name(p['neck_dim']) if p['arm'] == 'neck'
                  else rank_cell_name(p['dose']) if p['arm'] == 'rank'
                  else center_cell_name(p['dose']) if p['arm'] == 'center'
                  else cell_name(p['arm'], p['dose'], p['proj_dim']))
            if rt != c:
                bad.append((b, c, rt))
    return bad


if __name__ == '__main__':
    bad = _selftest()
    n_units = len(units())
    nc = len(cells(BACKBONES[0]))
    print(f'cells per backbone (active): {nc}  (arms: {ARMS_ACTIVE + ARMS_RIVAL} + base, '
          f'doses: {DOSES_ACTIVE})')
    print(f'total units (training runs): {n_units}')
    print(f'metric files per unit: 2 (val + test) | embeddings: 2 | protocol: fit on validation')
    print(f'total metric files: {n_units * 2}')
    print('round-trip test:', 'PASSED' if not bad else f'FAILED {bad[:5]}')
