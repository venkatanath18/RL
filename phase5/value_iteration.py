"""Value iteration for the four machine-replacement models."""


# ============================================================
# PARAMETERS
# ============================================================

TOTAL_STATES = 70
MAX_BUDGET = 12

MIN_COST = 990.0
MAX_COST = 9990.0
REPLACEMENT_COST = 9000.0

DISCOUNT_FACTOR = 0.7
TOLERANCE = 1e-9
MAX_ITERATIONS = 10000

REPLACE = 0
CONTINUE = 1

BASELINE = "baseline"
DEPLETING = "budget_depleting"
POSITIVE = "positive_cost"
POSITIVE_PLUS = "positive_cost_plus"

CASES = [BASELINE, DEPLETING, POSITIVE, POSITIVE_PLUS]

CASE_NAMES = {
    BASELINE: "BASELINE",
    DEPLETING: "BUDGET-DEPLETING MODEL",
    POSITIVE: "POSITIVE-COST MODEL",
    POSITIVE_PLUS: "POSITIVE-COST-PLUS MODEL",
}

STATES = list(range(1, TOTAL_STATES + 1))


# ============================================================
# COSTS AND TRANSITION PROBABILITIES
# ============================================================


def build_state_costs():
    costs = [0.0] * (TOTAL_STATES + 1)

    for state in STATES:
        costs[state] = (
            MIN_COST
            + (state - 1)
            * (MAX_COST - MIN_COST)
            / (TOTAL_STATES - 1)
        )

    return costs


def transition_probability(current_state, future_state):
    numerator = TOTAL_STATES + current_state - future_state
    denominator = (
        TOTAL_STATES * TOTAL_STATES
        + TOTAL_STATES * current_state
        - TOTAL_STATES * (TOTAL_STATES + 1) / 2
    )
    return numerator / denominator


def build_transition_probabilities():
    probabilities = [[0.0] * (TOTAL_STATES + 1)
                     for _ in range(TOTAL_STATES + 1)]

    for current_state in STATES:
        for future_state in STATES:
            probabilities[current_state][future_state] = (
                transition_probability(current_state, future_state)
            )

    return probabilities


STATE_COSTS = build_state_costs()
TRANSITION_PROBABILITIES = build_transition_probabilities()
REPLACE_TOTAL_COST = REPLACEMENT_COST + STATE_COSTS[1]


# ============================================================
# CONSTRAINT COST l(s,C)
# ============================================================


def constraint_cost(case, state):
    if case == BASELINE:
        return 0

    if case == DEPLETING:
        return 1 if state >= 35 else 0

    if case == POSITIVE:
        if state <= 20:
            return -1
        if state <= 34:
            return 0
        return 1

    if state <= 10:
        return -2
    if state <= 20:
        return -1
    if state <= 34:
        return 0
    if state <= 42:
        return 1
    return -1


def budget_values(case):
    if case == BASELINE:
        return [0]
    return list(range(MAX_BUDGET + 1))


def continue_budget(case, state, budget):
    if case == BASELINE:
        return 0

    new_budget = budget - constraint_cost(case, state)

    if new_budget < 0:
        return None

    return min(MAX_BUDGET, new_budget)


# ============================================================
# VALUE ITERATION
# ============================================================


def expected_value(values, current_state, budget):
    total = 0.0

    for future_state in STATES:
        total += (
            TRANSITION_PROBABILITIES[current_state][future_state]
            * values[future_state][budget]
        )

    return total


def action_costs(case, state, budget, values):
    # Replacement uses the state-1 transition distribution.
    # Replacement does not change the budget.
    replace_cost = (
        REPLACE_TOTAL_COST
        + DISCOUNT_FACTOR * expected_value(values, 1, budget)
    )

    next_budget = continue_budget(case, state, budget)

    if next_budget is None:
        continue_cost = float("inf")
    else:
        continue_cost = (
            STATE_COSTS[state]
            + DISCOUNT_FACTOR
            * expected_value(values, state, next_budget)
        )

    return replace_cost, continue_cost


def value_iteration(case):
    budgets = budget_values(case)
    values = [[0.0] * len(budgets) for _ in range(TOTAL_STATES + 1)]

    for iteration in range(1, MAX_ITERATIONS + 1):
        new_values = [[0.0] * len(budgets)
                      for _ in range(TOTAL_STATES + 1)]
        maximum_change = 0.0

        for state in STATES:
            for budget in budgets:
                replace_cost, continue_cost = action_costs(
                    case, state, budget, values
                )
                new_values[state][budget] = min(
                    replace_cost, continue_cost
                )
                maximum_change = max(
                    maximum_change,
                    abs(new_values[state][budget] - values[state][budget])
                )

        values = new_values

        if maximum_change < TOLERANCE:
            break

    policy = [[CONTINUE] * len(budgets)
              for _ in range(TOTAL_STATES + 1)]

    for state in STATES:
        for budget in budgets:
            replace_cost, continue_cost = action_costs(
                case, state, budget, values
            )
            policy[state][budget] = (
                REPLACE if replace_cost <= continue_cost else CONTINUE
            )

    return values, policy, iteration


# ============================================================
# CONSOLE OUTPUT
# ============================================================


def action_letter(action):
    return "R" if action == REPLACE else "C"


def print_policy(case, policy, iteration):
    print("\n" + "=" * 110)
    print(CASE_NAMES[case])
    print("=" * 110)

    if case == BASELINE:
        print("state | policy")
        print("------+-------")

        for state in STATES:
            print(f"{state:5d} |   {action_letter(policy[state][0])}")

    else:
        header = " | ".join(
            f"z={budget:2d}" for budget in range(MAX_BUDGET + 1)
        )
        print(f"state | l(s,C) | {header}")
        print("-" * 110)

        for state in STATES:
            actions = " | ".join(
                f"  {action_letter(policy[state][budget])} "
                for budget in range(MAX_BUDGET + 1)
            )
            print(
                f"{state:5d} | {constraint_cost(case, state):6d} | "
                f"{actions}"
            )

    print(f"Iterations: {iteration}")


def main():
    print("C = CONTINUE")
    print("R = REPLACE")
    print(f"States = {TOTAL_STATES}")
    print(f"Maximum budget = {MAX_BUDGET}")

    for case in CASES:
        values, policy, iteration = value_iteration(case)
        print_policy(case, policy, iteration)


if __name__ == "__main__":
    main()
