"""Field sales reps, weekly conversion, a coaching rollout, and a known true effect.

Each rep-week: n meetings, conversion prob sigmoid(base + rep skill + week-level luck/lead mix + season + coaching effect).
Paired counterfactual: the same meeting-level uniforms are used with and without coaching, so the true effect on the
coached rep-weeks is known exactly.
"""
import numpy as np

R, T, MAXN, PRE, POST = 300, 26, 40, 4, 4


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def simulate(seed, tau, trigger=False, capacity=0.5, coached_share=0.6, luck_sd=0.35):
    """tau: coaching effect in logit points (ramps in over 2 weeks).
    trigger=False: coaching starts at a random week for a random 60% of reps (waitlist style).
    trigger=True: a rep whose trailing 4-week rate is in the bottom quartile that week is offered coaching, capacity-limited."""
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 0.5, R)
    luck = rng.normal(0, luck_sd, (R, T))
    season = 0.15 * np.sin(2 * np.pi * np.arange(T) / 13) + 0.01 * np.arange(T)
    n = np.clip(rng.poisson(15, (R, T)), 3, MAXN)
    U = rng.random((R, T, MAXN))
    mask = np.arange(MAXN)[None, None, :] < n[:, :, None]
    base = -1.0 + skill[:, None] + luck + season[None, :]
    start = np.full(R, -1)
    pseudo = np.full(R, -1)           # first trigger week for reps who triggered but were not coached
    if not trigger:
        who = rng.random(R) < coached_share
        start[who] = rng.integers(PRE + 1, T - POST, who.sum())
    conv = np.zeros((R, T))
    conv_cf = np.zeros((R, T))
    for t in range(T):
        k = np.where((start >= 0) & (t >= start), t - start + 1, 0)
        eff = tau * np.minimum(1.0, k / 2.0)
        p1, p0 = sigmoid(base[:, t] + eff), sigmoid(base[:, t])
        conv[:, t] = ((U[:, t] < p1[:, None]) & mask[:, t]).sum(1)
        conv_cf[:, t] = ((U[:, t] < p0[:, None]) & mask[:, t]).sum(1)
        if trigger and PRE <= t < T - POST:
            rate = conv[:, max(0, t - PRE + 1):t + 1].sum(1) / n[:, max(0, t - PRE + 1):t + 1].sum(1)
            trig = (rate <= np.quantile(rate, 0.25)) & (start < 0) & (pseudo < 0)
            offered = trig & (rng.random(R) < capacity)
            start[offered] = t + 1
            pseudo[trig & ~offered] = t + 1
    return {"conv": conv, "n": n, "conv_cf": conv_cf, "start": start, "pseudo": pseudo}


def window(d, rows, g):
    """(post rate - pre rate) for each rep around its week g (pre = g-4..g-1, post = g..g+3)."""
    c, n = d["conv"], d["n"]
    pre = c[rows, g - PRE:g].sum(1) / n[rows, g - PRE:g].sum(1)
    post = c[rows, g:g + POST].sum(1) / n[rows, g:g + POST].sum(1)
    return pre, post


def true_effect(d):
    """Exact effect on the coached rep-weeks, relative to their counterfactual rate (post windows)."""
    rows = np.where(d["start"] >= 0)[0]
    a = b = 0.0
    for r in rows:
        g = d["start"][r]
        a += d["conv"][r, g:g + POST].sum()
        b += d["conv_cf"][r, g:g + POST].sum()
    return a / b - 1
