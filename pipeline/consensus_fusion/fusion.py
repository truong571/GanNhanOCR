"""Calibrated stacked fusion of correlated + independent channels.

With correlated voters, supervised stacking on a small gold set beats unsupervised
Dawid–Skene / majority (arXiv:2605.29800, arXiv:2601.22336): the combiner learns to
discount redundant channels because their features are collinear. We fit an
L2-regularised logistic regression (IRLS, from scratch — no sklearn) on the human-audit
labels, then isotonically calibrate its output so P is an honest probability of the
label being correct.

Handles missing channels (NaN, e.g. s3_cosine absent on GOLD-direct) by mean-imputation
plus a per-channel missing indicator, so a channel's absence is itself a feature.
"""
from __future__ import annotations

import numpy as np

__all__ = ["LogisticFuser", "IsotonicCalibrator", "roc_auc"]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def roc_auc(scores, y) -> float:
    """Rank-based ROC-AUC (Mann–Whitney). Returns 0.5 if a class is absent."""
    s = np.asarray(scores, float)
    y = np.asarray(y, int)
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    # tie-averaged ranks (Mann–Whitney U)
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    avg = np.empty(counts.size)
    start = 0
    for k, c in enumerate(counts):
        avg[k] = (start + 1 + start + c) / 2.0
        start += c
    ranks = avg[inv]
    n1, n0 = pos.size, neg.size
    r_pos = ranks[y == 1].sum()
    return float((r_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0))


class LogisticFuser:
    """L2-regularised logistic regression via IRLS with imputation + standardisation."""

    def __init__(self, l2: float = 1.0, add_missing_flags: bool = True,
                 max_iter: int = 100, tol: float = 1e-9):
        self.l2 = l2
        self.add_missing_flags = add_missing_flags
        self.max_iter = max_iter
        self.tol = tol
        self.feature_names_: list[str] = []

    # -- feature engineering ------------------------------------------------- #
    def _design(self, X, names, fit):
        X = np.asarray(X, float)
        n, d = X.shape
        if fit:
            self.impute_ = np.array([np.nanmean(X[:, j]) if np.any(~np.isnan(X[:, j]))
                                     else 0.0 for j in range(d)])
        Xi = X.copy()
        miss = np.isnan(Xi)
        for j in range(d):
            Xi[miss[:, j], j] = self.impute_[j]
        cols = [Xi]
        colnames = list(names)
        if self.add_missing_flags:
            # only for channels that actually have missingness in training
            if fit:
                self.miss_cols_ = [j for j in range(d) if miss[:, j].any()]
            flags = np.array([miss[:, j].astype(float) for j in self.miss_cols_]).T \
                if self.miss_cols_ else np.zeros((n, 0))
            if flags.size:
                cols.append(flags)
                colnames += [f"{names[j]}__missing" for j in self.miss_cols_]
        M = np.hstack(cols)
        if fit:
            self.mean_ = M.mean(axis=0)
            self.std_ = M.std(axis=0)
            self.std_[self.std_ == 0] = 1.0
            self.feature_names_ = colnames
        Ms = (M - self.mean_) / self.std_
        return np.hstack([np.ones((n, 1)), Ms])          # bias column first

    # -- fit / predict ------------------------------------------------------- #
    def fit(self, X, y, names=None):
        y = np.asarray(y, float)
        if names is None:
            names = [f"f{j}" for j in range(np.asarray(X).shape[1])]
        A = self._design(X, names, fit=True)
        n, p = A.shape
        w = np.zeros(p)
        reg = np.full(p, self.l2)
        reg[0] = 0.0                                     # never regularise the bias
        for _ in range(self.max_iter):
            eta = A @ w
            mu = _sigmoid(eta)
            W = np.clip(mu * (1 - mu), 1e-9, None)
            grad = A.T @ (mu - y) + reg * w
            H = (A * W[:, None]).T @ A + np.diag(reg) + 1e-8 * np.eye(p)
            step = np.linalg.solve(H, grad)
            w -= step
            if np.max(np.abs(step)) < self.tol:
                break
        self.coef_ = w
        return self

    def predict_proba(self, X, names=None):
        if names is None:
            names = [f"f{j}" for j in range(np.asarray(X).shape[1])]
        A = self._design(X, names, fit=False)
        return _sigmoid(A @ self.coef_)


class IsotonicCalibrator:
    """Monotone (PAV) mapping raw scores -> calibrated probabilities."""

    def fit(self, scores, y):
        s = np.asarray(scores, float)
        y = np.asarray(y, float)
        order = np.argsort(s, kind="mergesort")
        xs, ys = s[order], y[order]
        self.x_ = xs
        self.y_ = self._pav(ys)
        return self

    @staticmethod
    def _pav(y):
        # Pool-adjacent-violators: non-decreasing least-squares fit of y (ordered by x).
        blocks = []                       # [value, weight, count]
        for v in y:
            blocks.append([float(v), 1.0, 1])
            while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
                v2, w2, c2 = blocks.pop()
                v1, w1, c1 = blocks.pop()
                nw = w1 + w2
                blocks.append([(v1 * w1 + v2 * w2) / nw, nw, c1 + c2])
        out = []
        for v, _, c in blocks:
            out.extend([v] * c)
        return np.array(out)

    def transform(self, scores):
        s = np.asarray(scores, float)
        if self.x_.size == 0:
            return np.clip(s, 0, 1)
        return np.clip(np.interp(s, self.x_, self.y_), 0.0, 1.0)
