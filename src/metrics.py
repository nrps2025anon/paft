"""The 20 projection-quality measures and the four families they group into.

All 20 are always computed; there is no flag that turns any of them off, so a measure
cannot go silently missing. The families are local geometry, global geometry, label
structure and cluster-level structure.

Constants: K = 15 for the neighborhood-based measures, the R_NX curve is split at a fixed
K = 50, 50 triplets for the triplet accuracy, random state 0. Steadiness and cohesiveness
rest on random walks; they are run with 600 iterations and seeded from the training seed,
so a baseline and a method within one seed share the walks and the paired difference is
not dominated by sampling noise.
"""
import random
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import pairwise_distances, silhouette_score

K_NEIGHBORS = 15
N_TRIPLETS = 50
RANDOM_STATE = 0
QLG_SPLIT = 50
SC_ITERATIONS = 600

FAMILY = {
    'Local geometry': ['trustworthiness', 'continuity', 'auc_rnx', 'q_local', 'lcmc',
                       'mrre_high2low', 'mrre_low2high'],
    'Global geometry': ['pearson_dist_corr', 'shepard_goodness', 'normalized_stress',
                        'random_triplet_acc', 'q_global'],
    'Label structure': ['knn_accuracy', 'neighborhood_hit', 'silhouette', 'distance_consistency'],
    'Cluster-level': ['steadiness', 'cohesiveness', 'label_continuity', 'label_trustworthiness'],
}
METRICS = [m for ms in FAMILY.values() for m in ms]
LOWER = {'normalized_stress', 'mrre_high2low', 'mrre_low2high'}


def compute_geometry(X_high, X_low):
    X_high = np.ascontiguousarray(X_high, dtype=np.float64)
    X_low = np.ascontiguousarray(X_low, dtype=np.float64)
    Dh = pairwise_distances(X_high)
    Dl = pairwise_distances(X_low)
    Rh = np.argsort(np.argsort(Dh, axis=1), axis=1)
    Rl = np.argsort(np.argsort(Dl, axis=1), axis=1)
    N = X_high.shape[0]
    mask = ~np.eye(N, dtype=bool)
    Q = np.zeros((N - 1, N - 1), dtype=np.int64)
    np.add.at(Q, (Rh[mask] - 1, Rl[mask] - 1), 1)
    return dict(N=N, Dh=Dh, Dl=Dl, Rh=Rh, Rl=Rl, Q=Q,
                dh=pdist(X_high), dl=pdist(X_low))


def rnx_curve(G):
    Q, N = G['Q'], G['N']
    csum = Q.cumsum(0).cumsum(1)
    K = np.arange(1, N - 1)
    QNX = csum[K - 1, K - 1] / (K * N)
    RNX = ((N - 1) * QNX - K) / (N - 1 - K)
    return K, QNX, RNX


def _tc_penalty(R_set, R_pen, k, N):
    pen = 0.0
    for i in range(N):
        nb = np.where((R_set[i] >= 1) & (R_set[i] <= k))[0]
        rp = R_pen[i, nb]
        pen += np.sum(rp[rp > k] - k)
    norm = 2.0 / (N * k * (2 * N - 3 * k - 1))
    return float(1.0 - norm * pen)


def trustworthiness(G, k): return _tc_penalty(G['Rl'], G['Rh'], k, G['N'])
def continuity(G, k):      return _tc_penalty(G['Rh'], G['Rl'], k, G['N'])


def auc_rnx(G):
    K, _, RNX = rnx_curve(G)
    return float(np.sum(RNX / K) / np.sum(1.0 / K))


def normalized_stress(G):
    dh, dl = G['dh'], G['dl']
    alpha = np.dot(dh, dl) / (np.dot(dl, dl) + 1e-12)
    return float(np.sqrt(np.sum((dh - alpha * dl) ** 2) / (np.sum(dh ** 2) + 1e-12)))


def shepard_goodness(G):
    return float(spearmanr(G['dh'], G['dl']).correlation)


def neighborhood_hit(G, labels, k):
    Rl, lab = G['Rl'], np.asarray(labels)
    hits = [np.mean(lab[np.where((Rl[i] >= 1) & (Rl[i] <= k))[0]] == lab[i])
            for i in range(G['N'])]
    return float(np.mean(hits))


def knn_accuracy(G, labels, k):
    Rl, lab = G['Rl'], np.asarray(labels)
    correct = 0
    for i in range(G['N']):
        nb = np.where((Rl[i] >= 1) & (Rl[i] <= k))[0]
        vals, cnts = np.unique(lab[nb], return_counts=True)
        correct += (vals[np.argmax(cnts)] == lab[i])
    return float(correct / G['N'])


def q_local_global(G, split=QLG_SPLIT):
    """Split the R_NX curve at a fixed K = 50. Splitting at its own peak is unstable for PCA and
    sometimes undefined, and a split that moves with the run is not comparable across arms.
    """
    K, _, RNX = rnx_curve(G)
    kmax = split
    loc, glo = RNX[K <= kmax], RNX[K > kmax]
    return (float(np.mean(loc)) if loc.size else float('nan'),
            float(np.mean(glo)) if glo.size else float('nan'))


def lcmc(G, k):
    Q, N = G['Q'], G['N']
    csum = Q.cumsum(0).cumsum(1)
    QNX_k = csum[k - 1, k - 1] / (k * N)
    return float(QNX_k - k / (N - 1))


def mrre(G, k):
    Rh, Rl, N = G['Rh'], G['Rl'], G['N']
    s = np.arange(1, k + 1)
    C = N * np.sum(np.abs(N - 2 * s) / s)
    hl = lh = 0.0
    for i in range(N):
        jh = np.where((Rh[i] >= 1) & (Rh[i] <= k))[0]
        hl += np.sum(np.abs(Rh[i, jh] - Rl[i, jh]) / Rh[i, jh])
        jl = np.where((Rl[i] >= 1) & (Rl[i] <= k))[0]
        lh += np.sum(np.abs(Rh[i, jl] - Rl[i, jl]) / Rl[i, jl])
    return float(hl / C), float(lh / C)


def pearson_dist_corr(G):
    return float(pearsonr(G['dh'], G['dl'])[0])


def random_triplet_accuracy(G, n_triplets=N_TRIPLETS, seed=RANDOM_STATE):
    Dh, Dl, N = G['Dh'], G['Dl'], G['N']
    rng = np.random.default_rng(seed)
    correct = total = 0
    for i in range(N):
        others = np.delete(np.arange(N), i)
        b = rng.choice(others, n_triplets)
        c = rng.choice(others, n_triplets)
        valid = b != c
        agree = (Dh[i, b][valid] < Dh[i, c][valid]) == (Dl[i, b][valid] < Dl[i, c][valid])
        correct += int(np.sum(agree)); total += int(np.sum(valid))
    return float(correct / total)


def silhouette(X_low, labels):
    try:    return float(silhouette_score(X_low, labels))
    except Exception: return float('nan')


def distance_consistency(X_low, labels):
    lab = np.asarray(labels); classes = np.unique(lab)
    cents = np.array([X_low[lab == c].mean(0) for c in classes])
    nearest = classes[np.argmin(pairwise_distances(X_low, cents), axis=1)]
    return float(np.mean(nearest == lab))


def _zadu():
    """The measure library targets a newer Python; the enum it expects is shimmed in and the
    measure functions are called directly.
    """
    import enum
    if not hasattr(enum, 'StrEnum'):
        class StrEnum(str, enum.Enum):
            def __str__(self): return self.value
        enum.StrEnum = StrEnum
    from zadu.measures import steadiness_cohesiveness as sc
    from zadu.measures import label_trustworthiness_and_continuity as lt
    return sc, lt


_SC, _LT = _zadu()


def cluster_metrics(X_high, X_low, labels, k=K_NEIGHBORS, rng_seed=0):
    """Steadiness and cohesiveness. They rest on random walks and the library does not seed them,
    so the global seed is set from the training seed here. This fixes the sampling, not the
    definition of the measures.
    """
    np.random.seed(rng_seed)
    random.seed(rng_seed)
    hd = np.ascontiguousarray(X_high, np.float32)
    ld = np.ascontiguousarray(X_low, np.float32)
    lab = np.asarray(labels)
    out = {}
    out.update(_SC.measure(hd, ld, iteration=SC_ITERATIONS))
    out.update(_LT.measure(hd, ld, lab))
    return out


def evaluate(X_high, X_low, labels, k=K_NEIGHBORS, rng_seed=0):
    """All 20 measures as a flat dictionary. A missing measure raises rather than being omitted."""
    if hasattr(X_high, 'numpy'): X_high = X_high.numpy()
    if hasattr(X_low, 'numpy'):  X_low = X_low.numpy()
    X_high = np.asarray(X_high, np.float64)
    X_low = np.asarray(X_low, np.float64)
    labels = np.asarray(labels)
    G = compute_geometry(X_high, X_low)
    qloc, qglb = q_local_global(G)
    mrre_hl, mrre_lh = mrre(G, k)
    out = {
        'trustworthiness': trustworthiness(G, k),
        'continuity': continuity(G, k),
        'auc_rnx': auc_rnx(G),
        'normalized_stress': normalized_stress(G),
        'shepard_goodness': shepard_goodness(G),
        'neighborhood_hit': neighborhood_hit(G, labels, k),
        'knn_accuracy': knn_accuracy(G, labels, k),
        'q_local': qloc, 'q_global': qglb, 'lcmc': lcmc(G, k),
        'mrre_high2low': mrre_hl, 'mrre_low2high': mrre_lh,
        'pearson_dist_corr': pearson_dist_corr(G),
        'random_triplet_acc': random_triplet_accuracy(G),
        'silhouette': silhouette(X_low, labels),
        'distance_consistency': distance_consistency(X_low, labels),
    }
    out.update(cluster_metrics(X_high, X_low, labels, k, rng_seed=rng_seed))
    missing = [m for m in METRICS if m not in out]
    assert not missing, f'missing measures: {missing}'
    return {m: out[m] for m in METRICS}
