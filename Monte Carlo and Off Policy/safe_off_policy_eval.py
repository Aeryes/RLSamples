import numpy as np

# Define the target policy and behavior policy
def target_policy(action, state):
    # Target policy prefers action 1 with 80% probability
    return 0.8 if action == 1 else 0.2


def behavior_policy(action, state):
    # Behavior policy selects actions uniformly (50-50)
    return 0.5


# Simulate an episode under the behavior policy
def generate_episode(behavior_policy, num_steps=5):
    """Simulate an episode where the agent follows the behavior policy."""
    episode = []
    state = 0  # We'll keep the state fixed for simplicity
    for _ in range(num_steps):
        # Choose action based on behavior policy
        action = np.random.choice([0, 1], p=[0.5, 0.5])
        reward = np.random.randint(1, 11)  # Random reward between 1 and 10
        episode.append((state, action, reward))
    return episode


# Compute the importance sampling ratio
def importance_sampling_ratio(action, state):
    return target_policy(action, state) / behavior_policy(action, state)


# Safe off-policy evaluation with Chernoff-Hoeffding bound
def safe_off_policy_evaluation(episodes, delta, G_max):
    """Estimate a lower bound for V_pi using Chernoff-Hoeffding bounds."""
    n = len(episodes)  # Number of episodes
    total_weighted_returns = 0
    importance_weights = []

    # Loop over each episode
    for episode in episodes:
        G = 0  # Total return for this episode
        importance_weight = 1  # Cumulative importance sampling ratio

        for (state, action, reward) in episode:
            G += reward  # Sum rewards to compute G
            # Compute importance sampling ratio for the action
            importance_weight *= importance_sampling_ratio(action, state)

        # Accumulate weighted returns
        total_weighted_returns += importance_weight * G
        importance_weights.append(importance_weight)

    # Chernoff-Hoeffding confidence bound
    mean_importance_weighted_returns = total_weighted_returns / n
    bound_correction = G_max * np.sqrt(np.log(1 / delta) / (2 * n))

    # Lower bound on the value of the target policy
    V_pi_lower_bound = mean_importance_weighted_returns - bound_correction
    return V_pi_lower_bound

if __name__ == "__main__":
    # Simulation parameters
    num_episodes = 10
    num_steps_per_episode = 5
    delta = 0.05  # Confidence level (95% confidence interval)
    G_max = 10  # Max possible return in an episode (e.g., reward = 10 at each step)

    # Simulate episodes
    np.random.seed(42)  # Set seed for reproducibility
    episodes = [generate_episode(behavior_policy, num_steps_per_episode) for _ in range(num_episodes)]

    # Calculate the safe off-policy evaluation lower bound
    V_pi_lb = safe_off_policy_evaluation(episodes, delta, G_max)
    print(f"Estimated lower bound for the value of the target policy: {V_pi_lb:.2f}")
