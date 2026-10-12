# Phase 2: Learning the replacement policy

Phase 1 solved the machine-replacement problem with value iteration and policy
iteration. Those methods calculate expected future costs from the known
transition probabilities.

Phase 2 asks whether the replacement policy can instead be learned from sampled
machine transitions. The learner observes the current condition, selects either
REPLACE or CONTINUE, observes the next condition and monetary cost, and updates
its estimated cost for that state-action pair.

Action definitions in both programs:

- `0` = REPLACE the machine
- `1` = CONTINUE operating the machine

Because the table contains monetary costs, the preferred action is the action
with the smaller Q-value.

## Files

### `early_q_learning.py`

This is the first Q-learning prototype. It uses 15 condition states, a fixed
exploration setting and a learning rate based on the episode and step numbers.
It is retained to document the development path.

### `q_learning.py`

This is the refined implementation. It uses:

- epsilon-greedy exploration with decay;
- visit counts for every state-action pair;
- a decreasing learning rate based on those visit counts;
- the cost-minimizing Q-learning update; and
- a fixed random seed for reproducible runs.

The refined experiment has 50 condition states and a replacement cost of 900,
so its numerical configuration differs from the 15-state Phase 1 example. It
demonstrates the learning method rather than serving as an exact algorithm-to-
algorithm comparison with Phase 1.

## Run

```bash
python3 phase2/early_q_learning.py
python3 phase2/q_learning.py
```

The refined run performs 10 million sampled transitions and prints progress
every 2,000 episodes.
