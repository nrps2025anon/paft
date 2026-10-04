"""Stage A: train one cell, that is one (backbone, dataset, arm, seed).

The best epoch is chosen on validation accuracy and its state is restored at the end of
training, so validation and test representations are extracted in a single pass. Model
weights are not saved; what is stored per cell is the penultimate features, the bottleneck
features and the predictions for both splits.

The order of set_seed, model construction, device transfer, loader construction and
optimizer construction fixes the random number stream and must not be changed.
"""
import argparse, copy, hashlib, json, os, random, socket, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths as P, registry as R, data as D
from model import SweepNet
from objective import loss_terms

STAGE_FILES = {
    'train':   ['paths.py', 'registry.py', 'data.py', 'model.py', 'objective.py', 'train.py'],
    'analyze': ['paths.py', 'registry.py', 'metrics.py', 'analyze.py'],
}
STAGE_PROTO_KEYS = {
    'train':   ['recipe', 'val_split', 'determinism', 'regime', 'transforms'],
    'analyze': ['eval', 'n_test', 'n_val'],
}


def code_hash(stage='train'):
    h = hashlib.sha256()
    for f in STAGE_FILES[stage]:
        h.update(open(os.path.join(HERE, f), 'rb').read())
    sub = {k: R.PROTOCOL[k] for k in STAGE_PROTO_KEYS[stage] if k in R.PROTOCOL}
    h.update(json.dumps(sub, sort_keys=True).encode())
    return h.hexdigest()[:16]


def apply_determinism(proto, backbone):
    det = proto.get('determinism', {})
    if not det.get('by_backbone', {}).get(backbone, False):
        return 'kapali'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', det.get('cublas_workspace_config', ':4096:8'))
    if det.get('torch_use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True, warn_only=bool(det.get('warn_only', True)))
    return 'acik'


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model, loader, device):
    from sklearn.metrics import f1_score
    model.eval(); ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(device))[0].argmax(1).cpu()); ys.append(y)
    yt = torch.cat(ys).numpy(); yp = torch.cat(ps).numpy()
    return float((yt == yp).mean()), float(f1_score(yt, yp, average='macro'))


@torch.no_grad()
def extract_full(model, loader, device):
    """Extract one full split: penultimate features, labels, predictions and, where the cell has an
    auxiliary branch, the bottleneck features. No subsampling.
    """
    model.eval(); fs, ys, sm, ps, asm = [], [], [], [], []
    for x, y in loader:
        m, a, f, p = model(x.to(device))
        fs.append(f.cpu().numpy().astype(np.float32)); ys.append(y.numpy())
        sm.append(F.softmax(m, 1).cpu().numpy().astype(np.float16))
        if p is not None:
            ps.append(p.cpu().numpy().astype(np.float32))
            asm.append(F.softmax(a, 1).cpu().numpy().astype(np.float16))
    out = {'feat': np.concatenate(fs), 'labels': np.concatenate(ys),
           'softmax': np.concatenate(sm)}
    if ps:
        out['aux'] = np.concatenate(ps)
        out['aux_softmax'] = np.concatenate(asm)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', required=True, choices=R.BACKBONES)
    ap.add_argument('--dataset', required=True, choices=R.DATASETS)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()

    proto = R.PROTOCOL
    b, ds, cell, seed = a.backbone, a.dataset, a.cell, a.seed
    marker = P.done_marker(b, ds, cell, seed, 'train')
    if os.path.exists(marker) and not a.force:
        print(f'[skip] {b}/{ds}/{cell}/seed{seed} zaten tamam'); return

    det_state = apply_determinism(proto, b)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prm = R.cell_params(cell)
    rec = proto['recipe']
    max_epochs = rec['max_epochs']; patience = rec['early_stopping']['patience']
    batch = rec['batch']

    P.ensure_seed_tree(b, ds, cell, seed)
    print(f'[train] {b}/{ds}/{cell}/seed{seed} arm={prm["arm"]} dose={prm["dose"]:g} '
          f'd={prm["proj_dim"]} neck={prm.get("neck_dim")} aux={prm["aux"]} '
          f'determinizm={det_state} host={socket.gethostname()}', flush=True)

    full_train, test_ds, nc = D.make_datasets(ds)
    vs = proto['val_split']
    tr_idx, va_idx, val_ds = D.make_val_subset(full_train, ds, vs['fraction'], vs['seed'])
    assert len(va_idx) == R.N_VAL[ds], f'validation size does not match the protocol: {len(va_idx)} != {R.N_VAL[ds]}'
    assert len(test_ds) == R.N_TEST[ds], f'test size does not match the protocol: {len(test_ds)} != {R.N_TEST[ds]}'
    train_ds = Subset(full_train, tr_idx.tolist())
    val_dl = DataLoader(val_ds, 256, shuffle=False, num_workers=8, pin_memory=True)
    test_dl = DataLoader(test_ds, 256, shuffle=False, num_workers=8, pin_memory=True)
    t0 = time.time()

    set_seed(seed)
    model = SweepNet(P.BACKBONES[b], nc, prm['aux'], proj_dim=prm['proj_dim'],
                     neck_dim=prm.get('neck_dim'),
                     centers=(prm.get('w_center', 0) > 0)).to(device)
    train_dl = DataLoader(train_ds, batch, shuffle=True, num_workers=8, pin_memory=True)
    opt = torch.optim.Adam(model.parameters(), lr=rec['lr'])

    curve, diverged = [], False
    best = dict(val_acc=-1.0, epoch=-1, state=None)
    since = 0
    for ep in range(max_epochs):
        model.train(); agg = {}; nb = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device); opt.zero_grad()
            loss, comps = loss_terms(model, x, y, prm)
            if not torch.isfinite(loss):
                print(f'   [DIVERGED] ep{ep+1} non-finite loss', flush=True)
                diverged = True; break
            loss.backward(); opt.step()
            for k, v in comps.items(): agg[k] = agg.get(k, 0.0) + v
            nb += 1
        if diverged: break
        val_acc, val_f1 = evaluate(model, val_dl, device)
        terms = ' '.join(f'{k}={v/nb:.3f}' for k, v in agg.items())
        improved = val_acc > best['val_acc']
        if improved:
            best = dict(val_acc=val_acc, epoch=ep + 1,
                        state=copy.deepcopy(model.state_dict()))
            since = 0
        else:
            since += 1
        print(f'   ep{ep+1:02d} val_acc={val_acc:.4f} val_f1={val_f1:.4f} | {terms}'
              f'{"  *en iyi*" if improved else f"  (sabir {since}/{patience})"}', flush=True)
        curve.append(dict(seed=seed, unit='epoch', t=ep + 1, val_acc=val_acc, val_f1=val_f1,
                          best=bool(improved), **{k: v / nb for k, v in agg.items()}))
        if since >= patience:
            print(f'   [EARLY STOPPING] ep{ep+1}, en iyi ep{best["epoch"]} '
                  f'val_acc={best["val_acc"]:.4f}', flush=True)
            break
    t_train = time.time() - t0

    if diverged or best['state'] is None:
        diverged = True
        acc = f1 = float('nan')
    else:
        model.load_state_dict(best['state'])
        acc, f1 = evaluate(model, test_dl, device)
        for _sp, _dl in (('val', val_dl), ('test', test_dl)):
            _dst = P.reps(b, ds, cell, seed, _sp)
            _tmp = _dst + f'.tmp{os.getpid()}'
            with open(_tmp, 'wb') as _fh:
                np.savez_compressed(_fh, **extract_full(model, _dl, device))
            os.replace(_tmp, _dst)

    import pandas as pd
    pd.DataFrame(curve).to_csv(P.curve(b, ds, cell, seed), index=False)
    json.dump(dict(acc=acc, f1=f1, diverged=diverged, n_test=len(test_ds), n_classes=nc,
                   best_epoch=best['epoch'], best_val_acc=best['val_acc'],
                   epochs_run=len(curve)),
              open(P.accuracy(b, ds, cell, seed), 'w'), indent=2)
    import timm
    json.dump(dict(backbone=b, backbone_id=P.BACKBONES[b], dataset=ds, cell=cell, seed=seed,
                   params=prm, batch=batch, lr=rec['lr'],
                   max_epochs=max_epochs, patience=patience,
                   best_epoch=best['epoch'], best_val_acc=best['val_acc'],
                   epochs_run=len(curve), n_val=len(va_idx), n_train=len(train_ds),
                   feat_dim=int(model.feat_dim), n_test=len(test_ds),
                   n_classes=nc, diverged=diverged, determinism=det_state,
                   code_hash=code_hash('train'), host=socket.gethostname(),
                   train_seconds=round(t_train, 1),
                   versions=dict(torch=torch.__version__, timm=timm.__version__,
                                 numpy=np.__version__),
                   finished=time.strftime('%Y-%m-%dT%H:%M:%S')),
              open(P.manifest(b, ds, cell, seed), 'w'), indent=2)
    open(marker, 'w').write(code_hash('train') + '\n')
    print(f'[train] done ep{best["epoch"]}/{len(curve)} val={best["val_acc"]:.4f} '
          f'test_acc={acc:.4f} diverged={diverged} sure={t_train/60:.1f}dk -> '
          f'{P.seed_dir(b, ds, cell, seed)}', flush=True)


if __name__ == '__main__':
    main()
