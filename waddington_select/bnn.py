"""
bnn.py — GeneDisco's Bayesian MLP (docs/ml_engines_plan.md, phase 2 item 3).

Ported from keypaper/code/genedisco (models/pytorch_models.BayesianMLP, meta_models): one hidden layer of
16 units, dropout p=0.5 after it, a linear output; trained with ordinary dropout; predicted with
"consistent" MC dropout — K masks, each shared by every row of the batch — so the K sampled predictions
are K whole networks. The embedding acquisition rules use is the hidden layer before dropout.

Used by the faithful GeneDisco baseline (arms/genedisco_faithful_arm.py; MSE on the screen score, batch 64,
100 epochs, as GeneDisco) and by the `bnn` engine in our protocol (ml_models; weighted BCE on hit labels).
Everything runs on the CPU from explicit seeds, so a seed fixes the result.
"""

from __future__ import annotations

import numpy as np

HIDDEN = 16
P_DROP = 0.5
MC_SAMPLES = 100


def _net(d: int):
    import torch

    class BayesianMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(d, HIDDEN)
            self.fc2 = torch.nn.Linear(HIDDEN, 1)

        def forward(self, x):  # training: ordinary per-row dropout
            h = torch.nn.functional.dropout(self.fc1(x), P_DROP, self.training)
            return self.fc2(torch.relu(h)).squeeze(-1)

        def mc(self, x, masks):  # masks (K, HIDDEN), each shared by all rows -> outputs (K, N)
            h = torch.relu(self.fc1(x).unsqueeze(0) * masks.unsqueeze(1))
            return (h @ self.fc2.weight.T).squeeze(-1) + self.fc2.bias

    return BayesianMLP()


def fit(X, target, weight=None, loss: str = "mse", epochs: int = 100, batch: int = 64, lr: float = 1e-3,
        seed: int = 0):
    """Train on (X, target); `loss` "mse" (GeneDisco, regression) or "bce" (logits vs 0/1, weighted)."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
    yt = torch.tensor(np.asarray(target), dtype=torch.float32)
    wt = torch.tensor(np.ones(len(yt)) if weight is None else np.asarray(weight), dtype=torch.float32)
    net = _net(Xt.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    per_row = (torch.nn.MSELoss(reduction="none") if loss == "mse"
               else torch.nn.BCEWithLogitsLoss(reduction="none"))
    n = len(yt)
    net.train()
    for _ in range(epochs):
        perm = torch.as_tensor(rng.permutation(n))
        for k in range(0, n, batch):
            idx = perm[k:k + batch]
            opt.zero_grad()
            ((per_row(net(Xt[idx]), yt[idx]) * wt[idx]).sum() / wt[idx].sum()).backward()
            opt.step()
    net.eval()
    return net


def masks(seed: int, k: int = MC_SAMPLES):
    """K seeded dropout masks, scaled by 1/(1-p) as in training."""
    import torch
    g = torch.Generator().manual_seed(int(seed))
    return (torch.rand(k, HIDDEN, generator=g) >= P_DROP).float() / (1.0 - P_DROP)


def mc_predict(net, X, seed: int, k: int = MC_SAMPLES) -> np.ndarray:
    """(K, N) sampled outputs (raw: regression values, or logits for a BCE-trained net)."""
    import torch
    with torch.no_grad():
        return net.mc(torch.tensor(np.asarray(X), dtype=torch.float32), masks(seed, k)).numpy()


def embedding(net, X) -> np.ndarray:
    """The hidden layer before dropout (GeneDisco's get_embedding)."""
    import torch
    with torch.no_grad():
        return net.fc1(torch.tensor(np.asarray(X), dtype=torch.float32)).numpy()


def bim_distance(net, x: np.ndarray, seed: int, eps: float = 0.05, gamma: float = 0.35,
                 max_iter: int = 1000) -> float:
    """GeneDisco's adversarialBIM score for one gene: step the input along sign(d Var_MC / dx) until it has
    moved gamma * ||x||; return ||eta||^2 (a small value = the prediction is easily made uncertain).
    Fresh masks each step, as GeneDisco draws new samples on every call."""
    import torch
    nx = torch.tensor(np.asarray(x), dtype=torch.float32).unsqueeze(0)
    first = nx.clone()
    nx.requires_grad_()
    eta = torch.zeros_like(nx)
    it = 0
    while torch.linalg.norm(nx + eta - first) < gamma * torch.linalg.norm(first):
        if it >= max_iter:
            break
        out = net.mc(nx + eta, masks(seed * 100003 + it))
        out.var().backward()
        eta += eps * torch.sign(nx.grad.data)
        nx.grad.data.zero_()
        it += 1
    return float((eta * eta).sum())


def kmeanspp_seeds(E: np.ndarray, k: int, rng: np.random.Generator) -> list[int]:
    """BADGE's k-means++ seeding as GeneDisco implements it: start at the largest-norm row, then draw each
    next row with probability proportional to its squared distance to the nearest chosen one."""
    from sklearn.metrics import pairwise_distances
    chosen = [int(np.argmax(np.linalg.norm(E, axis=1)))]
    D = pairwise_distances(E, E[chosen]).ravel()
    while len(chosen) < min(k, len(E)):
        p = D ** 2
        if p.sum() <= 0:  # every row coincides with a chosen one; fill uniformly from the rest
            rest = np.setdiff1d(np.arange(len(E)), chosen)
            chosen.append(int(rng.choice(rest)))
        else:
            chosen.append(int(rng.choice(len(E), p=p / p.sum())))
        D = np.minimum(D, pairwise_distances(E, E[[chosen[-1]]]).ravel())
    return chosen


def kmeans_closest(E: np.ndarray, k: int, seed: int, rng: np.random.Generator) -> list[int]:
    """GeneDisco's kmeans rules: k-means (k-means++, n_init=10) on E, take the row nearest each centre.
    Centres can share a nearest row; the batch is topped up at random so it always has k genes."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances
    km = KMeans(n_clusters=min(k, len(E)), init="k-means++", n_init=10, random_state=seed).fit(E)
    picked = list(dict.fromkeys(pairwise_distances(E, km.cluster_centers_).argmin(axis=0).tolist()))
    for j in rng.permutation(len(E)):
        if len(picked) >= k:
            break
        if j not in picked:
            picked.append(int(j))
    return picked[:k]
