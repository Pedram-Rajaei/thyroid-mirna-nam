import math

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2


def t_score(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    c = np.unique(y)
    if len(c) != 2:
        raise ValueError('t_score needs a binary label vector')

    class0 = X[y == c[0]]
    class1 = X[y == c[1]]
    t = class0.mean(axis=0) - class1.mean(axis=0)
    se = np.sqrt(class0.var(axis=0) / len(class0) + class1.var(axis=0) / len(class1))
    F = np.divide(t, se, out=np.zeros_like(t), where=se > 0)
    return np.abs(F)


def t_score_select(X, y, k):
    score = t_score(X, y)
    return np.argsort(score)[::-1][:k]


def _prob(values):
    p = {}
    for v in values:
        p[v] = p.get(v, 0) + 1
    return {v: n / len(values) for v, n in p.items()}


def information_gain(X, y, n_bins=10):
    X = np.asarray(X, dtype=float)
    y = list(np.asarray(y))
    prob_y = _prob(y)
    H_y = -sum(p * math.log(p, 2) for p in prob_y.values())

    ig = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        x = list(pd.cut(X[:, i], n_bins, labels=False, duplicates='drop'))
        prob_x = _prob(x)
        prob_xy = _prob(list(zip(x, y)))
        H_y_given_x = 0.0
        for (xv, yv), p in prob_xy.items():
            H_y_given_x -= p * math.log(p / prob_x[xv], 2)
        ig[i] = H_y - H_y_given_x
    return ig


def info_gain_select(X, y, k, n_bins=10):
    ig = information_gain(X, y, n_bins)
    return np.argsort(ig)[::-1][:k]


def chi2_select(X, y, k):
    return SelectKBest(chi2, k=k).fit(X, y).get_support(indices=True)


def relieff_select(X, y, k, n_neighbors=100):
    from skrebate import ReliefF

    clf = ReliefF(n_features_to_select=k, n_neighbors=n_neighbors)
    clf.fit(np.asarray(X, dtype=float), np.asarray(y))
    return clf.top_features_[:k]


METHODS = {
    't_score': t_score_select,
    'info_gain': info_gain_select,
    'chi2': chi2_select,
    'relieff': relieff_select,
}


def select_features(df, y, method, k):
    """df: samples x miRNAs. Returns the names of the k selected miRNAs."""
    idx = METHODS[method](df.to_numpy(dtype=float), y, k)
    return list(df.columns[idx])
