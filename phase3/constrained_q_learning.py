"""Penalty-based, budget-augmented Q-learning for machine replacement."""

import argparse
import random


# ============================================================
# MACHINE REPLACEMENT + SAUTE-STYLE SAFETY AUGMENTATION
#
# action 0 = replace/reset
# action 1 = continue
#
# Safety interpretation used here:
#   - continuing in states >= SAFETY_START_STATE consumes 1 unit
#   - replacing consumes 0 safety units
#   - every replacement starts a fresh cycle with full budget
#   - trying to continue in a risky state with no budget left
#     gets a large finite penalty (Saute M_n approximation)
# ============================================================


# -----------------------------
# Original problem parameters
# -----------------------------

MAX_COST = 9990.0
MIN_COST = 990.0
RESET_COST = 9000.0

TOTAL_STATES = 50

DISCOUNT_FACTOR = 0.7

EPISODES = 100000
STEPS_PER_EPISODE = 500


# -----------------------------
# Q-learning parameters
# -----------------------------

EPSILON = 0.5
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.9995


# -----------------------------
# Safety parameters
# -----------------------------

# States 35, 36, ..., 50 are treated as the
# "high deterioration / risky-operation region".
SAFETY_START_STATE = 35

# A fresh machine cycle is allowed 2 risky CONTINUE decisions.
SAFETY_BUDGET = 2

# Finite penalty for an attempted budget overspend.
SAFETY_PENALTY = 50000.0


# -----------------------------
# State costs
# -----------------------------

state_costs = [0.0] * (TOTAL_STATES + 1)


def build_state_costs():
    slope = (MAX_COST - MIN_COST) / (TOTAL_STATES - 1)
    intercept = MIN_COST - slope

    costs = [0.0] * (TOTAL_STATES + 1)

    for state in range(1, TOTAL_STATES + 1):
        costs[state] = slope * state + intercept

    return costs


# -----------------------------
# Transition probabilities
# -----------------------------

def transition_probability(n, current_state, future_state):
    numerator = n + current_state - future_state

    denominator = (
        n * n
        + n * current_state
        - (n * (n + 1)) / 2
    )

    return numerator / denominator


STATES = list(range(1, TOTAL_STATES + 1))

# Precompute probabilities once so training is faster.
TRANSITION_WEIGHTS = {
    state: [
        transition_probability(
            TOTAL_STATES,
            state,
            future_state
        )
        for future_state in STATES
    ]
    for state in STATES
}


def sample_next_state(current_state):
    return random.choices(
        STATES,
        weights=TRANSITION_WEIGHTS[current_state],
        k=1
    )[0]


# ============================================================
# PART 1: UNCONSTRAINED Q-LEARNING
# ============================================================

Q_BASE = [
    [0.0, 0.0]
    for _ in range(TOTAL_STATES + 1)
]

VISITS_BASE = [
    [0, 0]
    for _ in range(TOTAL_STATES + 1)
]


def get_base_action(state, epsilon):

    if random.random() < epsilon:
        return random.choice([0, 1])

    if Q_BASE[state][0] < Q_BASE[state][1]:
        return 0

    if Q_BASE[state][1] < Q_BASE[state][0]:
        return 1

    return random.choice([0, 1])


def base_transition(state, action):

    # replace
    if action == 0:

        cost = RESET_COST + state_costs[1]

        # Replacement convention:
        # after replacement, transition as a fresh state-1 machine.
        next_state = sample_next_state(1)

    # continue
    else:

        cost = state_costs[state]

        next_state = sample_next_state(state)

    return next_state, cost


def train_unconstrained():

    epsilon = EPSILON

    for episode in range(EPISODES):

        state = 1

        for _ in range(STEPS_PER_EPISODE):

            action = get_base_action(
                state,
                epsilon
            )

            next_state, cost = base_transition(
                state,
                action
            )

            VISITS_BASE[state][action] += 1

            alpha = 1.0 / (
                VISITS_BASE[state][action] ** 0.6
            )

            # Cost-minimizing Q-learning:
            #
            # Q(s,a) <- Q(s,a)
            #          + alpha[
            #             cost + gamma min_a'Q(s',a')
            #             - Q(s,a)
            #          ]

            target = (
                cost
                + DISCOUNT_FACTOR
                * min(Q_BASE[next_state])
            )

            Q_BASE[state][action] += (
                alpha
                * (
                    target
                    - Q_BASE[state][action]
                )
            )

            state = next_state

        epsilon = max(
            EPSILON_MIN,
            epsilon * EPSILON_DECAY
        )


def get_base_policy():

    policy = []

    for state in range(1, TOTAL_STATES + 1):

        if Q_BASE[state][0] < Q_BASE[state][1]:
            action = 0
        else:
            action = 1

        policy.append(action)

    return policy


# ============================================================
# PART 2: SAUTE-STYLE SAFETY-AUGMENTED Q-LEARNING
# ============================================================

# Q_SAFE[state][remaining_budget][action]
#
# This is the main Saute modification:
#
# ordinary state:
#       s
#
# augmented state:
#       (s, z)
#
# where z is the remaining safety budget.

Q_SAFE = [
    [
        [0.0, 0.0]
        for _ in range(SAFETY_BUDGET + 1)
    ]
    for _ in range(TOTAL_STATES + 1)
]


VISITS_SAFE = [
    [
        [0, 0]
        for _ in range(SAFETY_BUDGET + 1)
    ]
    for _ in range(TOTAL_STATES + 1)
]


def safety_cost(state, action):

    # Replacing does not consume safety budget.
    if action == 0:
        return 0

    # Continuing in a high-deterioration state
    # consumes one safety unit.
    if state >= SAFETY_START_STATE:
        return 1

    return 0


def get_safe_action(state, budget, epsilon):

    if random.random() < epsilon:
        return random.choice([0, 1])

    q_reset = Q_SAFE[state][budget][0]
    q_continue = Q_SAFE[state][budget][1]

    if q_reset < q_continue:
        return 0

    if q_continue < q_reset:
        return 1

    return random.choice([0, 1])


def safe_transition(state, budget, action):

    if not 0 <= budget <= SAFETY_BUDGET:
        raise ValueError("budget must be between 0 and SAFETY_BUDGET")

    # --------------------------------------------------------
    # ACTION 0: REPLACE
    #
    # Replacement ends the current machine cycle.
    # New machine => reset physical condition convention
    # and reset the safety budget.
    # --------------------------------------------------------

    if action == 0:

        cost = RESET_COST + state_costs[1]

        next_state = sample_next_state(1)

        next_budget = SAFETY_BUDGET

        violated = False

        return (
            next_state,
            next_budget,
            cost,
            violated
        )


    # --------------------------------------------------------
    # ACTION 1: CONTINUE
    # --------------------------------------------------------

    constraint_cost = safety_cost(
        state,
        action
    )

    next_state = sample_next_state(state)

    # Enough safety budget is available.
    if constraint_cost <= budget:

        next_budget = (
            budget
            - constraint_cost
        )

        cost = state_costs[state]

        violated = False


    # Continuing would push the remaining budget below zero.
    #
    # Record the violation before storing a clipped budget.
    # This finite penalty does not prevent unsafe exploration.
    else:

        next_budget = 0

        cost = SAFETY_PENALTY

        violated = True


    return (
        next_state,
        next_budget,
        cost,
        violated
    )


def train_safe():

    epsilon = EPSILON

    violations = 0

    for episode in range(EPISODES):

        # Simulator random starts for augmented state coverage
        state = random.randint(1, TOTAL_STATES)
        budget = random.randint(0, SAFETY_BUDGET)

        for _ in range(STEPS_PER_EPISODE):

            action = get_safe_action(
                state,
                budget,
                epsilon
            )

            (
                next_state,
                next_budget,
                cost,
                violated
            ) = safe_transition(
                state,
                budget,
                action
            )

            if violated:
                violations += 1


            VISITS_SAFE[state][budget][action] += 1

            alpha = 1.0 / (
                VISITS_SAFE[state][budget][action] ** 0.6
            )


            next_best_value = min(
                Q_SAFE[next_state][next_budget]
            )


            target = (
                cost
                + DISCOUNT_FACTOR
                * next_best_value
            )


            Q_SAFE[state][budget][action] += (
                alpha
                * (
                    target
                    - Q_SAFE[state][budget][action]
                )
            )


            state = next_state
            budget = next_budget


        epsilon = max(
            EPSILON_MIN,
            epsilon * EPSILON_DECAY
        )


    return violations


def get_safe_policy(budget):

    if not 0 <= budget <= SAFETY_BUDGET:
        raise ValueError("budget must be between 0 and SAFETY_BUDGET")

    policy = []

    for state in range(1, TOTAL_STATES + 1):

        if (
            Q_SAFE[state][budget][0]
            <
            Q_SAFE[state][budget][1]
        ):
            action = 0
        else:
            action = 1

        policy.append(action)

    return policy


# -----------------------------
# Threshold helper
# -----------------------------

def replacement_regions(policy):
    regions = []
    start = None
    for state, action in enumerate([*policy, 1], start=1):
        if action == 0 and start is None:
            start = state
        elif action != 0 and start is not None:
            regions.append((start, state - 1))
            start = None
    return regions


def first_reset_state(policy):

    for state, action in enumerate(
        policy,
        start=1
    ):

        if action == 0:
            return state

    return None


# -----------------------------
# Results
# -----------------------------

def print_results(violations):

    print()
    print("=" * 70)
    print("ACTION DEFINITIONS")
    print("=" * 70)

    print("0 = REPLACE")
    print("1 = CONTINUE")


    # --------------------------------
    # Baseline
    # --------------------------------

    base_policy = get_base_policy()

    print("Replacement regions (inclusive):", replacement_regions(base_policy))

    base_threshold = first_reset_state(
        base_policy
    )


    print()
    print("=" * 70)
    print("UNCONSTRAINED POLICY")
    print("=" * 70)

    print(
        "First replacement state:",
        base_threshold
    )

    print(
        "Policy:",
        base_policy
    )


    # --------------------------------
    # Safety augmented
    # --------------------------------

    print()
    print("=" * 70)
    print("PENALTY-BASED BUDGET-AUGMENTED POLICY")
    print("=" * 70)

    print(
        "Safety region: states",
        SAFETY_START_STATE,
        "to",
        TOTAL_STATES
    )

    print(
        "Initial safety budget:",
        SAFETY_BUDGET
    )

    print(
        "Meaning: each CONTINUE in the safety region "
        "uses one safety unit."
    )

    print(
        "Replacement resets budget to",
        SAFETY_BUDGET
    )

    print(
        "Violation penalty n =",
        SAFETY_PENALTY
    )


    print()
    print(
        "Budget-dependent replacement thresholds:"
    )


    # Print from full budget down to zero.
    for budget in range(
        SAFETY_BUDGET,
        -1,
        -1
    ):

        policy = get_safe_policy(
            budget
        )

        threshold = first_reset_state(
            policy
        )

        print(
            f"remaining budget z = {budget}: "
            f"first REPLACE state = {threshold}; "
            f"replacement regions = {replacement_regions(policy)}"
        )


    print()
    print(
        "Number of safety violations encountered "
        "during exploratory training:",
        violations
    )


    # --------------------------------
    # Policy details around the
    # interesting high-deterioration area
    # --------------------------------

    start_print = max(
        1,
        SAFETY_START_STATE - 5
    )


    print()
    print("=" * 70)
    print(
        f"POLICY DETAILS: STATES "
        f"{start_print}-{TOTAL_STATES}"
    )
    print("=" * 70)


    header = "state | baseline"

    for budget in range(
        SAFETY_BUDGET,
        -1,
        -1
    ):
        header += f" | z={budget}"

    print(header)
    print("-" * len(header))


    for state in range(
        start_print,
        TOTAL_STATES + 1
    ):

        row = (
            f"{state:5d} | "
            f"{base_policy[state - 1]:8d}"
        )

        for budget in range(
            SAFETY_BUDGET,
            -1,
            -1
        ):

            safe_policy = get_safe_policy(
                budget
            )

            row += (
                f" | "
                f"{safe_policy[state - 1]:3d}"
            )

        print(row)


# -----------------------------
# Main
# -----------------------------

def positive_integer(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main():
    global state_costs, EPISODES, STEPS_PER_EPISODE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=positive_integer, default=EPISODES)
    parser.add_argument("--steps-per-episode", type=positive_integer,
                        default=STEPS_PER_EPISODE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    EPISODES = args.episodes
    STEPS_PER_EPISODE = args.steps_per_episode
    state_costs = build_state_costs()

    random.seed(args.seed)
    print("Training unconstrained Q-learning...", flush=True)
    train_unconstrained()

    # Reproducible streams; different actions consume random draws differently.
    random.seed(args.seed)
    print("Training budget-augmented Q-learning...", flush=True)
    violations = train_safe()
    print_results(violations)

    for budget in range(SAFETY_BUDGET, -1, -1):
        print(
            f"state=46, z={budget}: "
            f"visits={VISITS_SAFE[46][budget]}, Q={Q_SAFE[46][budget]}"
        )


if __name__ == "__main__":
    main()
