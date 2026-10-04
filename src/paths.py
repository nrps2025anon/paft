"""The single addressing layer: every output path is built here.

Two rules follow from it. The aggregation layer never scans the file system; the registry
produces the expected units and their paths are computed here and read one by one. And one
file is one (backbone, dataset, cell, seed, split, pass, n) address whose name carries
every parameter, so a run cannot silently overwrite another.

Layout under results/:

  results/{backbone}/{dataset}/{cell}/seed{S}/
      manifest.json                config, code hash, duration, validation size
      train/curve.csv              per-epoch curve
      train/accuracy.json          test accuracy and macro-F1, best validation accuracy
      reps/{val,test}.npz          penultimate features, predictions, bottleneck features
      dr/{split}.npz               the four 2-D embeddings, full split
      dr/{split}_aux.npz           the same fitted on the bottleneck representation
      metrics/{split}_n{N}.csv     four projections x 20 measures at the measured n
      .done_train, .done_analyze_* completion markers
"""
import os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.environ.get('PAFT_RESULTS', os.path.join(ROOT, 'results'))

BACKBONES = {'r18': 'resnet18.a1_in1k',
             'r50': 'resnet50.a1_in1k',
             'vit': 'vit_small_patch16_224.augreg_in1k'}
DATASETS = ['svhn', 'stl10', 'cifar100', 'tinyimagenet', 'cub200']
DRS = ['pca', 'tsne', 'umap', 'pacmap']
DR_LABEL = {'pca': 'PCA', 'tsne': 't-SNE', 'umap': 'UMAP', 'pacmap': 'PaCMAP'}
SPLITS = ['val', 'test']

ARMS_OURS = ['kl_cka_nbr', 'kl_cka', 'kl_nbr', 'cka_nbr', 'kl', 'cka', 'nbr']
ARMS_RIVAL = ['trank']
DOSE = ['001', '005', '01', '025', '05', '1', '2', '5']
CELL_RE = re.compile(r'^(?P<arm>kl_cka_nbr|kl_cka|kl_nbr|cka_nbr|trank|kl|cka|nbr)'
                     r'_(?P<dose>001|005|01|025|05|1|2|5)_d(?P<d>16|64)$')
CELL_NECK_RE = re.compile(r'^neck_d(?P<nd>2|16|64|128|256|512|1024)$')
CELL_RANK_RE = re.compile(r'^rank_(?P<dose>0001|0005|001|0025|005|01)$')
CELL_CENTER_RE = re.compile(r'^center_(?P<dose>00003|0001|0003|001|003|01)$')
BASE = 'base'


def is_cell(name):
    return (name == BASE or bool(CELL_RE.match(name))
            or bool(CELL_NECK_RE.match(name)) or bool(CELL_RANK_RE.match(name))
            or bool(CELL_CENTER_RE.match(name)))


def cell_dir(backbone, dataset, cell):
    assert backbone in BACKBONES, f'unknown backbone: {backbone}'
    assert dataset in DATASETS, f'unknown dataset: {dataset}'
    assert is_cell(cell), f'cell name does not match the grammar: {cell}'
    return os.path.join(RESULTS, backbone, dataset, cell)


def seed_dir(backbone, dataset, cell, seed):
    assert isinstance(seed, int) and seed >= 0, f'seed must be an integer: {seed}'
    return os.path.join(cell_dir(backbone, dataset, cell), f'seed{seed}')


def _p(b, d, c, s, *parts):
    return os.path.join(seed_dir(b, d, c, s), *parts)


def manifest(b, d, c, s):   return _p(b, d, c, s, 'manifest.json')
def curve(b, d, c, s):      return _p(b, d, c, s, 'train', 'curve.csv')
def accuracy(b, d, c, s):   return _p(b, d, c, s, 'train', 'accuracy.json')


def reps(b, d, c, s, split):
    assert split in SPLITS, f'unknown split: {split}'
    return _p(b, d, c, s, 'reps', f'{split}.npz')


def projections(b, d, c, s, split, rep='feat'):
    """The four 2-D embeddings in one archive, keyed by projection, for the whole split. The file
    carries no n: the projection is fitted on validation and the full split is transformed,
    while the measures are taken at the n below.
    """
    assert split in SPLITS, f'unknown split: {split}'
    assert rep in ('feat', 'aux'), f'unknown representation: {rep}'
    return _p(b, d, c, s, 'dr', f'{split}.npz' if rep == 'feat' else f'{split}_aux.npz')


def metrics(b, d, c, s, split, n, rep='feat'):
    """Four projections x 20 measures in one file, named by the n the measures were computed at.
    Validation uses the full split; test is capped at 10000 points.
    """
    assert split in SPLITS, f'unknown split: {split}'
    assert rep in ('feat', 'aux'), f'unknown representation: {rep}'
    tag = '' if rep == 'feat' else '_aux'
    return _p(b, d, c, s, 'metrics', f'{split}{tag}_n{int(n):05d}.csv')


def done_marker(b, d, c, s, stage):
    return _p(b, d, c, s, f'.done_{stage}')


def ensure_seed_tree(b, d, c, s):
    base = seed_dir(b, d, c, s)
    for sub in ('train', 'reps', 'dr', 'metrics'):
        os.makedirs(os.path.join(base, sub), exist_ok=True)
    return base
