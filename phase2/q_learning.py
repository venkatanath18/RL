"""Refined Q-learning implementation for machine replacement."""

import random


MAX_COST = 999.0
MIN_COST = 99.0
RESET_COST = 900.0
TOTAL_STATES = 50
DISCOUNT_FACTOR = 0.7

EPISODES = 20000
STEPS_PER_EPISODE = 500

EPSILON = 0.5
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9995

# action 0 = replace; action 1 = continue
Q = [[0.0, 0.0] for _ in range(TOTAL_STATES + 1)]
visit_count = [[0, 0] for _ in range(TOTAL_STATES + 1)]
state_costs = [0.0] * (TOTAL_STATES + 1)


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


STATES = list(range(1, TOTAL_STATES + 1))

# The transition law is fixed, so calculate its weights once rather than on
# every one of the 10 million sampled steps.
TRANSITION_WEIGHTS = {
    state: [
        transition_probability(TOTAL_STATES, state, future_state)
        for future_state in STATES
    ]
    for state in STATES
}


def sample_next_state(current_state):
    return random.choices(
        STATES,
        weights=TRANSITION_WEIGHTS[current_state],
        k=1,
    )[0]


def next_state_cost(state, action):
    if action == 0:
        cost = RESET_COST + state_costs[1]
        next_state = sample_next_state(1)
    else:
        cost = state_costs[state]
        next_state = sample_next_state(state)
    return next_state, cost


def get_action(state, epsilon):
    if random.random() < epsilon:
        return random.choice([0, 1])
    if Q[state][0] < Q[state][1]:
        return 0
    if Q[state][1] < Q[state][0]:
        return 1
    return random.choice([0, 1])


def get_learning_rate(state, action):
    visit_count[state][action] += 1
    return 1.0 / (visit_count[state][action] ** 0.6)


def generate_episode(epsilon):
    state = 1
    for _ in range(STEPS_PER_EPISODE):
        action = get_action(state, epsilon)
        next_state, cost = next_state_cost(state, action)
        alpha = get_learning_rate(state, action)
        next_best_value = min(Q[next_state])
        target = cost + DISCOUNT_FACTOR * next_best_value
        Q[state][action] += alpha * (target - Q[state][action])
        state = next_state


def get_policy():
    policy = []
    for state in range(1, TOTAL_STATES + 1):
        policy.append(0 if Q[state][0] < Q[state][1] else 1)
    return policy


def print_results():
    print("\nAction definitions: 0 = REPLACE, 1 = CONTINUE")
    print("\nState costs:")
    for state in range(1, TOTAL_STATES + 1):
        print(f"State {state}: {state_costs[state]:.4f}")

    print("\nLearned Q values:")
    for state in range(1, TOTAL_STATES + 1):
        print(
            f"State {state:2d}: Q(replace) = {Q[state][0]:10.4f}, "
            f"Q(continue) = {Q[state][1]:10.4f}"
        )

    policy = get_policy()
    print("\nLearned policy:")
    print(policy)
    for state, action in enumerate(policy, start=1):
        label = "REPLACE" if action == 0 else "CONTINUE"
        print(f"State {state:2d}: {label}")


def main():
    global state_costs
    random.seed(42)
    state_costs = build_state_costs()
    epsilon = EPSILON

    for episode in range(EPISODES):
        generate_episode(epsilon)
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        if (episode + 1) % 2000 == 0:
            print(
                f"Episode {episode + 1:5d}, epsilon = {epsilon:.4f}, "
                f"policy = {get_policy()}"
            )

    print_results()


if __name__ == "__main__":
    main()
