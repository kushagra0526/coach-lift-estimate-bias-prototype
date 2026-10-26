import json

import numpy as np

from . import est, sim

TAU = 0.25   # true coaching effect in logit points when coaching works (about +15% relative conversion in the first 4 weeks)


def one(seed, tau, trigger):
    d = sim.simulate(seed, tau, trigger=trigger)
    truth = sim.true_effect(d)
    out = {"truth": truth}
    for name, fn in (("pre/post of coached reps", est.naive_pre_post), ("coached vs never coached", est.coached_vs_not),
                     ("diff-in-diff vs never coached", est.did_never_treated), ("trigger-matched controls", est.trigger_matched)):
        v = fn(d)
        lo, hi = est.boot(fn, d, B=40, seed=seed) if name in ("diff-in-diff vs never coached", "trigger-matched controls") else (np.nan, np.nan)
        out[name] = (v, lo, hi)
    return out


def score(seeds, trigger):
    res = {}
    for s in seeds:
        for tau in (0.0, TAU):
            r = one(s * 2 + (tau > 0), tau, trigger)
            for k, v in r.items():
                if k == "truth":
                    continue
                e = res.setdefault((k, tau), {"est": [], "truth": [], "sig": []})
                e["est"].append(v[0])
                e["truth"].append(r["truth"])
                e["sig"].append(bool(v[1] > 0) if not np.isnan(v[1]) else None)
    out = {}
    for (k, tau), e in res.items():
        est_, tr = np.array(e["est"], float), np.array(e["truth"])
        ok = ~np.isnan(est_)
        sig = [x for x in e["sig"] if x is not None]
        out.setdefault(k, {})[f"tau={tau}"] = {"mean_est_pct": round(100 * float(np.nanmean(est_)), 1), "mean_true_pct": round(100 * float(tr[ok].mean()), 1),
                                              "bias_pts": round(100 * float(np.mean(est_[ok] - tr[ok])), 1),
                                              "claims_lift_pct_of_worlds": round(100 * float(np.mean(sig)), 1) if sig else None, "worlds": int(ok.sum())}
    return out


def main():
    res = {"dev_random_rollout": score(range(0, 40), False), "held_out_struggle_triggered": score(range(1000, 1040), True)}
    json.dump(res, open("out/result.json", "w"), indent=1)
    for k, v in res.items():
        print(k)
        for m, d in v.items():
            print("  ", m)
            for t, x in d.items():
                print("     ", t, x)


if __name__ == "__main__":
    main()
