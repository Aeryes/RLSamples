# Off policy sample with importance sampling.
import numpy as np

# Step 1: Define policies
def target_policy(action, state):
    # Example: target policy favors action 1 over 0 with 80% probability
    return 0.8 if action == 1 else 0.2


def behavior_policy(action, state):
    # Example: behavior policy is more exploratory, chooses both actions equally
    return 0.5


# Step 2: Simulate episodes
def generate_episode(behavior_policy, num_steps=5):
    """Simulate an episode following the behavior policy."""
    episode = []
    state = 0  # We keep the state fixed for simplicity
    for _ in range(num_steps):
        # Randomly pick an action using the behavior policy
        action = np.random.choice([0, 1])  # Action 0 or 1
        reward = np.random.randint(1, 11)  # Random reward between 1 and 10
        episode.append((state, action, reward))
    return episode


# Step 3: Importance Sampling
def importance_sampling(episodes):
    """Estimate value function using ordinary importance sampling."""
    total_weighted_returns = 0
    total_importance_weights = 0
    num_episodes = len(episodes)

    for episode in episodes:
        importance_weight = 1
        G = 0  # Return G for this episode

        for (state, action, reward) in episode:
            G += reward
            # Compute importance ratio
            rho = target_policy(action, state) / behavior_policy(action, state)
            importance_weight *= rho  # Multiply importance ratios for this episode

        # Add weighted return to total
        total_weighted_returns += importance_weight * G
        total_importance_weights += importance_weight

    # Step 4: Return estimated value V_pi
    V_pi = total_weighted_returns / total_importance_weights if total_importance_weights != 0 else 0
    return V_pi

if __name__ == "__main__":
    # Step 5: Simulate and run the calculation
    np.random.seed(42)  # For reproducibility

    # Simulate multiple episodes following the behavior policy
    num_episodes = 10
    episodes = [generate_episode(behavior_policy) for _ in range(num_episodes)]

    # Estimate value function using ordinary importance sampling
    estimated_value = importance_sampling(episodes)
    print(f"Estimated value of target policy V_pi: {estimated_value:.2f}")
