# coachlift: how much of "coached reps convert X% better" is coaching?

COACH's launch post says its AI coaching is "lifting conversion rates by 32% on average" with the top 5 in-person sales companies in France, and that each rep starts the day with a brief on their weakest point (https://www.ycombinator.com/launches/SBP-coach-the-ai-coach-for-in-person-sales-reps). I don't know how that 32% was measured. The usual ways of measuring a coaching lift are biased in a known direction when the people who get coached are not random, for example when a rep is offered coaching after a bad few weeks. This is a simulator with a known true effect that shows how much.

## The setup
`coachlift/sim.py`: 300 reps over 26 weeks, ~15 meetings a week, conversion prob = sigmoid(base + rep skill + weekly luck/lead mix + season + coaching effect), where the coaching effect ramps in over 2 weeks. Meeting-level random numbers are shared with and without coaching, so the true effect on the coached rep-weeks is known exactly.
Two rollouts: (1) **random**: 60% of reps start coaching in a random week; (2) **struggle-triggered**: when a rep's trailing 4-week rate is in the bottom quartile they are offered coaching (capacity: half of them get it, the rest are untouched).
`coachlift/est.py`: four estimators of the relative lift in the 4 weeks after coaching starts: pre/post of coached reps; coached vs never-coached; difference-in-differences against never-coached reps (calendar-matched, with a 95% bootstrap over reps); and **trigger-matched controls**: coached reps' pre/post change compared with the pre/post change of reps who hit the same trigger and were not coached.

## Results (known truth; 40 worlds per row at true effect 0 and at +0.25 logit, about +15% relative)
Dev: random rollout, seeds 0-39. Held-out: struggle-triggered rollout, seeds 1000-1039 (the rollout rule was fixed before the first run).

| Held-out, struggle-triggered | true lift 0%: estimate | true lift +16.6%: estimate | claims a lift at true 0% |
|---|---|---|---|
| pre/post of coached reps | **+30.3%** | **+51.6%** | - |
| coached vs never coached | -26.5% | -10.6% | - |
| diff-in-diff vs never coached | **+34.0%** | **+55.2%** | 100% of worlds |
| trigger-matched controls | -1.2% | +16.5% | 5.0% |

Dev (random rollout, true lift 0% and +14.4%): pre/post -0.1% and +14.8%; coached vs never-coached -0.3% and +16.1%; diff-in-diff -0.2% and +14.9% (claims a lift in 7.5% of zero-effect worlds, 100% at +14.4%). Trigger-matched controls need a trigger rule, so they do not apply to a random rollout.

- With a random rollout, three of the methods are fine. With a struggle-triggered one, the two most common are badly wrong in opposite directions: pre/post and diff-in-diff read +30 to +55% from regression to the mean (the pre window contains the slump that caused the trigger), and coached-vs-not reads negative (the coached are the weak reps).
- Matching on the trigger removes it (-1.2% at zero, +16.5% at +16.6%), and detects the real effect in 77.5% of worlds at 300 reps and 4 post weeks.

## Read this before believing the table
The simulator, the trigger and the estimators are mine, and the trigger-matched fix needs reps who were triggered but not coached, which may not exist in a real rollout (if everyone triggered is coached, only a staggered or randomized start can identify the effect). Real conversion has long sales cycles, rep churn, lead mix by region (the "Texas 2x California" kind of comparison on COACH's page has the same problem) and effects that decay. I do not know how COACH measured its 32%; nothing here uses COACH data or systems.

## Run it (3 commands)
```
pip install numpy pytest
python -m pytest -q tests
python -m coachlift.run
```
numpy only. 4 tests.

Unofficial prototype by Kushagra Pratap Singh. Not affiliated with COACH.
