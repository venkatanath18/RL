"""Q-learning for the same four models used by value_iteration.py."""

import random

from value_iteration import (
    BASELINE,
    CASES,
    CASE_NAMES,
    CONTINUE,
    DISCOUNT_FACTOR,
    MAX_BUDGET,
    REPLACE,
    REPLACE_TOTAL_COST,
    STATE_COSTS,
    STATES,
    TOTAL_STATES,
    TRANSITION_PROBABILITIES,
    action_letter,
    budget_values,
    constraint_cost,
    continue_budget,
    value_iteration,
)


# ============================================================
# Q-LEARNING PARAMETERS
# ============================================================

SEED = 42
STEPS_PER_EPISODE = 100
MAX_EPISODES = 100000
CHECK_EVERY = 2000

MINIMUM_VISITS = 200
MINIMUM_POLICY_AGREEMENT = 0.98
MAXIMUM_POLICY_CHANGE = 0.02
REQUIRED_PASSES = 3

ALPHA_EXPONENT = 0.8


# ============================================================
# TRANSITION SAMPLE
# ============================================================


def sample_next_state(current_state, rng):
    return rng.choices(
        STATES,
        weights=TRANSITION_PROBABILITIES[current_state][1:],
        k=1,
    )[0]


def take_action(case, state, budget, action, rng):
    if action == REPLACE:
        next_state = sample_next_state(1, rng)
        next_budget = budget
        cost = REPLACE_TOTAL_COST
    else:
        next_state = sample_next_state(state, rng)
        next_budget = continue_budget(case, state, budget)
        cost = STATE_COSTS[state]

    return next_state, next_budget, cost


# ============================================================
# POLICY CHECKS
# ============================================================


def get_policy(q_values, budgets):
    policy = [[CONTINUE] * len(budgets)
              for _ in range(TOTAL_STATES + 1)]

    for state in STATES:
        for budget in budgets:
            if q_values[state][budget][REPLACE] <= q_values[state][budget][CONTINUE]:
                policy[state][budget] = REPLACE
            else:
                policy[state][budget] = CONTINUE

    return policy


def policy_agreement(policy, exact_policy, budgets):
    equal = 0
    total = TOTAL_STATES * len(budgets)

    for state in STATES:
        for budget in budgets:
            if policy[state][budget] == exact_policy[state][budget]:
                equal += 1

    return equal / total


def policy_change(policy, old_policy, budgets):
    if old_policy is None:
        return 1.0

    changed = 0
    total = TOTAL_STATES * len(budgets)

    for state in STATES:
        for budget in budgets:
            if policy[state][budget] != old_policy[state][budget]:
                changed += 1

    return changed / total


def minimum_visits(visits, case, budgets):
    minimum = float("inf")

    for state in STATES:
        for budget in budgets:
            minimum = min(minimum, visits[state][budget][REPLACE])

            if continue_budget(case, state, budget) is not None:
                minimum = min(minimum, visits[state][budget][CONTINUE])

    return int(minimum)


# ============================================================
# Q-LEARNING
# ============================================================


def q_learning(case):
    rng = random.Random(SEED)
    budgets = budget_values(case)

    q_values = [
        [[0.0, 0.0] for _ in budgets]
        for _ in range(TOTAL_STATES + 1)
    ]
    visits = [
        [[0, 0] for _ in budgets]
        for _ in range(TOTAL_STATES + 1)
    ]

    # An infeasible CONTINUE action is never selected.
    for state in STATES:
        for budget in budgets:
            if continue_budget(case, state, budget) is None:
                q_values[state][budget][CONTINUE] = float("inf")

    exact_policy = value_iteration(case)[1]
    previous_policy = None
    consecutive_passes = 0

    for episode in range(1, MAX_EPISODES + 1):
        # Random state-budget-action sampling gives every feasible pair visits.
        for _ in range(STEPS_PER_EPISODE):
            state = rng.choice(STATES)
            budget = rng.choice(budgets)

            if continue_budget(case, state, budget) is None:
                action = REPLACE
            else:
                action = rng.choice([REPLACE, CONTINUE])

            next_state, next_budget, cost = take_action(
                case, state, budget, action, rng
            )

            visits[state][budget][action] += 1
            alpha = visits[state][budget][action] ** (-ALPHA_EXPONENT)

            target = (
                cost
                + DISCOUNT_FACTOR
                * min(q_values[next_state][next_budget])
            )

            q_values[state][budget][action] += (
                alpha
                * (target - q_values[state][budget][action])
            )

        if episode % CHECK_EVERY == 0:
            policy = get_policy(q_values, budgets)
            agreement = policy_agreement(policy, exact_policy, budgets)
            change = policy_change(policy, previous_policy, budgets)
            visits_minimum = minimum_visits(visits, case, budgets)

            passed = (
                visits_minimum >= MINIMUM_VISITS
                and agreement >= MINIMUM_POLICY_AGREEMENT
                and change <= MAXIMUM_POLICY_CHANGE
            )

            if passed:
                consecutive_passes += 1
            else:
                consecutive_passes = 0

            print(
                f"{CASE_NAMES[case]:28s} "
                f"episode={episode:6d}  "
                f"agreement={agreement:7.2%}  "
                f"change={change:7.2%}  "
                f"passes={consecutive_passes}"
            )

            previous_policy = policy

            if consecutive_passes == REQUIRED_PASSES:
                return q_values, policy, episode, agreement

    policy = get_policy(q_values, budgets)
    agreement = policy_agreement(policy, exact_policy, budgets)
    return q_values, policy, MAX_EPISODES, agreement


# ============================================================
# OUTPUT
# ============================================================


def print_learned_policy(case, policy, episodes, agreement):
    print("\n" + "=" * 110)
    print(f"Q-LEARNING: {CASE_NAMES[case]}")
    print(f"Episodes: {episodes}")
    print(f"Agreement with value iteration: {agreement:.2%}")
    print("=" * 110)

    if case == BASELINE:
        print("state | policy")

        for state in STATES:
            print(f"{state:5d} |   {action_letter(policy[state][0])}")

    else:
        header = " | ".join(
            f"z={budget:2d}" for budget in range(MAX_BUDGET + 1)
        )
        print(f"state | l(s,C) | {header}")

        for state in STATES:
            actions = " | ".join(
                f"  {action_letter(policy[state][budget])} "
                for budget in range(MAX_BUDGET + 1)
            )
            print(
                f"{state:5d} | {constraint_cost(case, state):6d} | "
                f"{actions}"
            )


def main():
    print("C = CONTINUE")
    print("R = REPLACE")
    print(f"Seed = {SEED}")

    for case in CASES:
        q_values, policy, episodes, agreement = q_learning(case)
        print_learned_policy(case, policy, episodes, agreement)


if __name__ == "__main__":
    main()
