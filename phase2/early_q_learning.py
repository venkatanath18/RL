"""First Q-learning prototype for the machine-replacement problem."""

import random


MAX_COST = 999.0
MIN_COST = 99.0
RESET_COST = 990.0
ITERATIONS = 5000
TOTAL_STATES = 15
LEARNING_FACTOR = 0.5
DISCOUNT_FACTOR = 0.7
GREEDY_FACTOR = 0.5

current_best_action = [0] * (TOTAL_STATES + 1)
optimal_values = [[0.0] * 2 for _ in range(TOTAL_STATES + 1)]
state_costs = [0.0] * (TOTAL_STATES + 1)


def get_action(state):
    return random.choices(
        [1 - current_best_action[state], current_best_action[state]],
        [GREEDY_FACTOR / 2, 1 - GREEDY_FACTOR / 2],
        k=1,
    )[0]


def build_state_costs():
    slope = (MAX_COST - MIN_COST) / (TOTAL_STATES - 1)
    intercept = MIN_COST - slope
    costs = [0.0] * (TOTAL_STATES + 1)
    for state in range(1, TOTAL_STATES + 1):
        costs[state] = slope * state + intercept
    return costs


def transition_probability(n, current_state, future_state):
    numerator = n + current_state - future_state
    denominator = n * n + n * current_state - (n * (n + 1)) / 2
    return numerator / denominator


def sample_next_state(current_state):
    states = list(range(1, TOTAL_STATES + 1))
    probabilities = [
        transition_probability(TOTAL_STATES, current_state, future_state)
        for future_state in states
    ]
    return random.choices(states, weights=probabilities, k=1)[0]


def next_state_cost(state, action):
    if action == 0:
        cost = RESET_COST + state_costs[1]
        next_state = sample_next_state(1)
    else:
        cost = state_costs[state]
        next_state = sample_next_state(state)
    return next_state, cost


def update_best_action(state):
    if optimal_values[state][0] >= optimal_values[state][1]:
        current_best_action[state] = 1
    else:
        current_best_action[state] = 0


def generate_episode(iteration_count):
    state = 1
    for step in range(500):
        action = get_action(state)
        next_state, cost = next_state_cost(state, action)
        learning_rate = 1 / ((step + 1) * (iteration_count + 1))
        target = (
            cost
            + DISCOUNT_FACTOR
            * optimal_values[next_state][current_best_action[next_state]]
        )
        optimal_values[state][action] += learning_rate * (
            target - optimal_values[state][action]
        )
        update_best_action(state)
        state = next_state


def main():
    global state_costs
    random.seed(42)
    state_costs = build_state_costs()
    for iteration in range(ITERATIONS):
        generate_episode(iteration)

    print(current_best_action)
    print("Q(state, action), where action 0 = replace and action 1 = continue:")
    for state in range(1, TOTAL_STATES + 1):
        print(f"Q({state}, 0) = {optimal_values[state][0]}")
        print(f"Q({state}, 1) = {optimal_values[state][1]}")


if __name__ == "__main__":
    main()
