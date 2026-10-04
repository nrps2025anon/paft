"""Training objective: cross-entropy plus one coefficient times the selected terms.

    ours     : CE(main) + a * [ KL(stopgrad(main) || aux) + (1 - linearCKA(f, p))
                                + neighborhood(f, p) ]
    top-d    : CE(main) + a * topd(f; d)
    nuclear  : CE(main) + a * nuclear(f)
    center   : CE(main) + a * center(f, y)
    baseline : CE(main)

f is the penultimate representation and p the bottleneck. Dropping terms from the first
line gives the objective-term ablations. The penalties live on different scales: the
nuclear norm is around 9 in training while top-d is in [0, 1], so the same coefficient is
far more aggressive for the nuclear arm and its dose ladder is scaled down accordingly.
"""
import math
import torch
import torch.nn.functional as F


def linear_cka(a, b):
    def cen(M):
        n = M.shape[0]
        H = torch.eye(n, device=M.device) - torch.ones(n, n, device=M.device) / n
        return H @ M @ H
    K, L = cen(a @ a.t()), cen(b @ b.t())
    return (K * L).sum() / (torch.sqrt((K * K).sum() * (L * L).sum()) + 1e-6)


def nuclear_penalty(f):
    """Nuclear norm over Frobenius norm: convex, scale-free, and blind to both labels and the
    target dimension. It ranges from 1 for a rank-one batch up to the square root of the
    smaller batch dimension.
    """
    fc = f - f.mean(0, keepdim=True)
    s = torch.linalg.svdvals(fc.float())
    return s.sum() / s.norm().clamp_min(1e-12)


def topd_penalty(f, target_d):
    """One minus the variance fraction in the top-d directions. Scale-free, so shrinking the
    features cannot game it.
    """
    fc = f - f.mean(0, keepdim=True)
    s = torch.linalg.svdvals(fc.float())
    e = s ** 2
    k = min(int(target_d), e.numel())
    return 1.0 - e.topk(k).values.sum() / e.sum().clamp_min(1e-12)


def nbr_penalty(f, p, perplexity=30.0):
    """Neighborhood preservation, the parametric t-SNE loss in its raw form. The neighborhood
    distribution in the penultimate space is detached, so the term pulls the bottleneck
    towards that neighborhood instead of collapsing the penultimate representation to make
    the term cheap.
    """
    n = f.shape[0]
    if n < 4 or p is None:
        return f.new_zeros(())
    mask = ~torch.eye(n, dtype=torch.bool, device=f.device)
    with torch.no_grad():
        d2 = torch.cdist(f, f).pow_(2)
        target = math.log(min(perplexity, n - 1))
        beta = torch.ones(n, device=f.device)
        lo = torch.full((n,), 1e-12, device=f.device)
        hi = torch.full((n,), float('inf'), device=f.device)
        for _ in range(32):
            lg = (-beta.unsqueeze(1) * d2).masked_fill(~mask, float('-inf'))
            lp = torch.log_softmax(lg, 1)
            H = -(lp.exp() * torch.where(mask, lp, torch.zeros_like(lp))).sum(1)
            gt = H > target
            lo = torch.where(gt, beta, lo)
            hi = torch.where(gt, hi, beta)
            beta = torch.where(torch.isinf(hi), beta * 2, (lo + hi) / 2)
        Pj = torch.softmax((-beta.unsqueeze(1) * d2).masked_fill(~mask, float('-inf')), 1)
        Pm = ((Pj + Pj.t()) / (2.0 * n)).clamp_min(1e-12)
    num = (1.0 / (1.0 + torch.cdist(p, p) ** 2)) * mask
    Q = (num / num.sum().clamp_min(1e-12)).clamp_min(1e-12)
    return (Pm[mask] * (Pm[mask].log() - Q[mask].log())).sum()


def center_term(model, f, y, alpha=0.5):
    """Center loss with the update rule of the original paper. The class centers are buffers rather
    than parameters and live across batches, which is what makes the arm usable on the
    datasets with 100 and 200 classes. One declared deviation: the loss is averaged over the
    minibatch rather than summed, so it shares the normalization of cross-entropy.
    """
    c = model.centers
    ct = 0.5 * ((f - c[y].detach()) ** 2).sum(1).mean()
    with torch.no_grad():
        diff = c[y] - f.detach()
        cnt = torch.zeros(c.shape[0], device=f.device, dtype=c.dtype)
        cnt.index_add_(0, y, torch.ones_like(y, dtype=c.dtype))
        num = torch.zeros_like(c)
        num.index_add_(0, y, diff)
        c -= alpha * num / (1.0 + cnt).unsqueeze(1)
    return ct


def loss_terms(model, x, y, prm):
    """Total loss and the per-term breakdown, from the parameters of one cell."""
    m, a, f, p = model(x)
    ce = F.cross_entropy(m, y); loss = ce
    comps = {'ce': ce.item()}
    if prm['w_kl'] > 0:
        kl = F.kl_div(F.log_softmax(a, 1), F.softmax(m.detach(), 1), reduction='batchmean')
        loss = loss + prm['w_kl'] * kl; comps['kl'] = kl.item()
    if prm['w_cka'] > 0:
        cka_pen = 1 - linear_cka(f, p)
        loss = loss + prm['w_cka'] * cka_pen; comps['cka'] = cka_pen.item()
    if prm['w_nbr'] > 0:
        nb = nbr_penalty(f, p)
        loss = loss + prm['w_nbr'] * nb; comps['nbr'] = nb.item()
    if prm.get('w_center', 0) > 0:
        ct = center_term(model, f, y)
        loss = loss + prm['w_center'] * ct; comps['center'] = ct.item()
    if prm['w_rank'] > 0:
        rk = (nuclear_penalty(f) if prm.get('rank_mode') == 'nuclear'
              else topd_penalty(f, prm['proj_dim']))
        loss = loss + prm['w_rank'] * rk; comps['rank'] = rk.item()
    return loss, comps
