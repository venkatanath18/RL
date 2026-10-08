import matplotlib.pyplot as plt


MAX_COST = 999.0
MIN_COST = 99.0
RESET_COST = 990.0
TOTAL_STATES = 15
ITERATIONS = 500
DISCOUNT_FACTOR = 0.7


def transition_probability(n, current_state, future_state):
    numerator = n + current_state - future_state
    denominator = n * n + n * current_state - (n * (n + 1)) / 2
    return numerator / denominator


def build_state_costs():
    slope = (MAX_COST - MIN_COST) / (TOTAL_STATES - 1)
    intercept = MIN_COST - slope
    costs = [0.0] * (TOTAL_STATES + 1)

    for state in range(1, TOTAL_STATES + 1):
        costs[state] = slope * state + intercept

    return costs


def discounted_future_cost(values, current_state):
    expected_value = 0.0

    for future_state in range(1, TOTAL_STATES + 1):
        expected_value += (
            transition_probability(TOTAL_STATES, current_state, future_state)
            * values[future_state]
        )

    return DISCOUNT_FACTOR * expected_value


def value_iteration(state_costs):
    values = [0.0] * (TOTAL_STATES + 1)

    for _ in range(ITERATIONS):
        new_values = [0.0] * (TOTAL_STATES + 1)

        reset_cost = (
            RESET_COST
            + state_costs[0]
            + discounted_future_cost(values, current_state=0)
        )

        for state in range(1, TOTAL_STATES + 1):
            continue_cost = (
                state_costs[state]
                + discounted_future_cost(values, current_state=state)
            )
            new_values[state] = min(reset_cost, continue_cost)

        values = new_values

    return values


def optimal_policy(state_costs, values):
    policy = [0] * (TOTAL_STATES + 1)

    reset_cost = (
        RESET_COST
        + state_costs[0]
        + discounted_future_cost(values, current_state=0)
    )

    for state in range(1, TOTAL_STATES + 1):
        continue_cost = (
            state_costs[state]
            + discounted_future_cost(values, current_state=state)
        )

        # 0 = reset, 1 = continue
        policy[state] = 0 if reset_cost <= continue_cost else 1

    return policy


def policy_iteration(state_costs):
    values = [0.0] * (TOTAL_STATES + 1)
    policy = [0] * (TOTAL_STATES + 1)

    for _ in range(ITERATIONS):
        for _ in range(ITERATIONS):
            new_values = [0.0] * (TOTAL_STATES + 1)

            for state in range(1, TOTAL_STATES + 1):
                if policy[state] == 0:
                    new_values[state] = (
                        RESET_COST
                        + state_costs[0]
                        + discounted_future_cost(values, current_state=0)
                    )
                else:
                    new_values[state] = (
                        state_costs[state]
                        + discounted_future_cost(values, current_state=state)
                    )

            values = new_values

        policy = optimal_policy(state_costs, values)

    return values, policy


def plot_results(values, policy):
    states = list(range(1, TOTAL_STATES + 1))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(states, values[1:], marker="o", linewidth=2)
    ax1.set_ylabel("V(s)")
    ax1.set_title("Optimal Value Function")
    ax1.grid(True, alpha=0.3)

    ax2.step(states, policy[1:], where="mid", linewidth=2)
    ax2.scatter(states, policy[1:], zorder=3)
    ax2.set_xlabel("State s")
    ax2.set_ylabel("Policy")
    ax2.set_title("Optimal Policy")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Reset", "Continue"])
    ax2.set_ylim(-0.2, 1.2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_results(state_costs, values, policy):
    print("State costs:")
    for state in range(1, TOTAL_STATES + 1):
        print(f"s={state:2d}: {state_costs[state]:10.4f}")

    print("\nOptimal value and policy:")
    print(f"{'s':>3} {'V(s)':>14} {'policy':>12}")
    print("-" * 33)
    for state in range(1, TOTAL_STATES + 1):
        action = "reset" if policy[state] == 0 else "continue"
        print(f"{state:3d} {values[state]:14.4f} {action:>12}")


def main():
    state_costs = build_state_costs()
    values, policy = policy_iteration(state_costs)

    print_results(state_costs, values, policy)
    plot_results(values, policy)


if __name__ == "__main__":
    main()
