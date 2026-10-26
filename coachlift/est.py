"""Four ways to read "coached reps convert X% better". All return a relative lift (post vs expected), in fraction."""
import numpy as np

from .sim import POST, PRE, T


def _changes(d, rows, weeks):
    pre = np.empty(len(rows))
    post = np.empty(len(rows))
    for i, (r, g) in enumerate(zip(rows, weeks)):
        a, b = _one(d, r, g)
        pre[i], post[i] = a, b
    return pre, post


def _one(d, r, g):
    c, n = d["conv"], d["n"]
    return c[r, g - PRE:g].sum() / n[r, g - PRE:g].sum(), c[r, g:g + POST].sum() / n[r, g:g + POST].sum()


def naive_pre_post(d, idx=None):
    rows = np.where(d["start"] >= 0)[0] if idx is None else idx[d["start"][idx] >= 0]
    pre, post = _changes(d, rows, d["start"][rows])
    return post.mean() / pre.mean() - 1


def coached_vs_not(d, idx=None):
    idx = np.arange(len(d["start"])) if idx is None else idx
    cw = slice(T - 8, T)
    rate = lambda r: d["conv"][r, cw].sum() / d["n"][r, cw].sum()
    t = idx[d["start"][idx] >= 0]
    u = idx[d["start"][idx] < 0]
    return np.mean([rate(r) for r in t]) / np.mean([rate(r) for r in u]) - 1


def did_never_treated(d, idx=None):
    """Coached reps' pre->post change over the same calendar window's change for reps never coached (pooled rates)."""
    idx = np.arange(len(d["start"])) if idx is None else idx
    t = idx[d["start"][idx] >= 0]
    u = idx[d["start"][idx] < 0]
    cc = np.concatenate([[0.0], np.cumsum(d["conv"][u].sum(0))])
    nn = np.concatenate([[0.0], np.cumsum(d["n"][u].sum(0))])
    rate = lambda lo, hi: (cc[hi] - cc[lo]) / (nn[hi] - nn[lo])
    ch, base = [], []
    for r in t:
        g = d["start"][r]
        ch.append(_one(d, r, g))
        base.append((rate(g - PRE, g), rate(g, g + POST)))
    ch, base = np.array(ch), np.array(base)
    return (ch[:, 1].mean() / ch[:, 0].mean()) / (base[:, 1].mean() / base[:, 0].mean()) - 1


def trigger_matched(d, idx=None):
    """Compare coached reps' pre->post change with that of reps who hit the same trigger but were not coached (same week-window logic)."""
    idx = np.arange(len(d["start"])) if idx is None else idx
    t = idx[d["start"][idx] >= 0]
    p = idx[d["pseudo"][idx] >= 0]
    if len(p) < 5:
        return np.nan
    a, b = _changes(d, t, d["start"][t])
    pa, pb = _changes(d, p, d["pseudo"][p])
    return (b.mean() / a.mean()) / (pb.mean() / pa.mean()) - 1


def boot(fn, d, B=60, seed=0):
    rng = np.random.default_rng(seed)
    n = len(d["start"])
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        v = fn(d, idx)
        if not np.isnan(v):
            vals.append(v)
    return np.quantile(vals, [0.025, 0.975]) if len(vals) > 20 else (np.nan, np.nan)
