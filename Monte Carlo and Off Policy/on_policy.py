# On policy sample with
import numpy as np

# Step 1: Define the policy
def policy(action, state):
    # A simple deterministic policy: 80% chance of choosing action 1 in any state
    return 0.8 if action == 1 else 0.2


# Step 2: Simulate episodes following the policy
def generate_episode(policy, num_steps=5):
    """Simulate an episode following the given policy."""
    episode = []
    state = 0  # Keep the state fixed for simplicity
    for _ in range(num_steps):
        # Choose an action according to the policy
        action = np.random.choice([0, 1], p=[0.2, 0.8])  # Follows policy (80% chance of action 1)
        reward = np.random.randint(1, 11)  # Random reward between 1 and 10
        episode.append((state, action, reward))
    return episode


# Step 3: On-policy value estimation
def on_policy_value_estimation(episodes):
    """Estimate value function V(s) using on-policy Monte Carlo and Off Policy method."""
    state_value_returns = []

    for episode in episodes:
        G = 0  # Total return
        for (state, action, reward) in episode:
            G += reward  # Sum up the total reward

        # Append total return from this episode to the list of returns
        state_value_returns.append(G)

    # Step 4: Return the average return as the estimated value
    V_pi = np.mean(state_value_returns)
    return V_pi

if __name__ == "__main__":
    # Step 5: Simulate and calculate the value estimate
    np.random.seed(42)  # For reproducibility

    # Generate multiple episodes by following the policy
    num_episodes = 10
    episodes = [generate_episode(policy) for _ in range(num_episodes)]

    # Estimate value function using on-policy Monte Carlo and Off Policy
    estimated_value = on_policy_value_estimation(episodes)
    print(f"Estimated value of the policy V_pi: {estimated_value:.2f}")
