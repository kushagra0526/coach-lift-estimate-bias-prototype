import numpy as np

from coachlift import est, sim


def test_reproducible_and_true_effect_zero_without_coaching_effect():
    a, b = sim.simulate(3, 0.0), sim.simulate(3, 0.0)
    assert np.array_equal(a["conv"], b["conv"])
    assert sim.true_effect(a) == 0.0


def test_random_rollout_pre_post_and_did_are_roughly_right():
    out = [(est.naive_pre_post(d), est.did_never_treated(d), sim.true_effect(d)) for d in (sim.simulate(s, 0.25) for s in range(10))]
    for pp, did, tr in out:
        assert abs(did - tr) < 0.12


def test_struggle_triggered_rollout_inflates_pre_post_with_no_effect():
    vals = [est.naive_pre_post(sim.simulate(s, 0.0, trigger=True)) for s in range(10)]
    assert np.mean(vals) > 0.15


def test_trigger_matched_controls_remove_the_inflation():
    vals = [est.trigger_matched(sim.simulate(s, 0.0, trigger=True)) for s in range(20)]
    assert abs(np.nanmean(vals)) < 0.08
