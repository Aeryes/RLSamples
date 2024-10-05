# Cliff world sample with Q-learning and SARSA.
import numpy as np

# Cliff environment setup
class CliffWalkingEnv:
    def __init__(self, grid_height=4, grid_width=12):
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.start_state = (3, 0)
        self.goal_state = (3, 11)
        self.cliff = [(3, i) for i in range(1, 11)]
        self.state = self.start_state

    def reset(self):
        self.state = self.start_state
        return self.state

    def step(self, action):
        """Move the agent in the grid based on the action."""
        row, col = self.state
        if action == 0:  # Up
            row = max(row - 1, 0)
        elif action == 1:  # Down
            row = min(row + 1, self.grid_height - 1)
        elif action == 2:  # Left
            col = max(col - 1, 0)
        elif action == 3:  # Right
            col = min(col + 1, self.grid_width - 1)

        next_state = (row, col)

        # Check if the agent fell off the cliff
        if next_state in self.cliff:
            reward = -100
            next_state = self.start_state  # Reset to start
        elif next_state == self.goal_state:
            reward = 0  # Reached goal
        else:
            reward = -1  # Regular move

        self.state = next_state
        return next_state, reward, next_state == self.goal_state

    def get_num_states(self):
        return self.grid_height * self.grid_width

    def get_num_actions(self):
        return 4  # Up, Down, Left, Right


# Helper function: choose an action based on epsilon-greedy policy
def epsilon_greedy_action(Q, state, num_actions, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(0, num_actions)  # Explore: random action
    else:
        return np.argmax(Q[state])  # Exploit: choose the best action


# On-policy SARSA
def sarsa(env, num_episodes=500, alpha=0.5, gamma=0.9, epsilon=0.1):
    Q = np.zeros((env.get_num_states(), env.get_num_actions()))
    num_actions = env.get_num_actions()

    for episode in range(num_episodes):
        state = env.reset()
        state_idx = state[0] * env.grid_width + state[1]
        action = epsilon_greedy_action(Q, state_idx, num_actions, epsilon)

        done = False
        while not done:
            next_state, reward, done = env.step(action)
            next_state_idx = next_state[0] * env.grid_width + next_state[1]
            next_action = epsilon_greedy_action(Q, next_state_idx, num_actions, epsilon)

            # SARSA update
            Q[state_idx, action] += alpha * (
                    reward + gamma * Q[next_state_idx, next_action] - Q[state_idx, action]
            )

            state_idx = next_state_idx
            action = next_action

    return Q


# Off-policy Q-Learning
def q_learning(env, num_episodes=500, alpha=0.5, gamma=0.9, epsilon=0.1):
    Q = np.zeros((env.get_num_states(), env.get_num_actions()))
    num_actions = env.get_num_actions()

    for episode in range(num_episodes):
        state = env.reset()
        state_idx = state[0] * env.grid_width + state[1]

        done = False
        while not done:
            action = epsilon_greedy_action(Q, state_idx, num_actions, epsilon)
            next_state, reward, done = env.step(action)
            next_state_idx = next_state[0] * env.grid_width + next_state[1]

            # Q-Learning update (off-policy)
            Q[state_idx, action] += alpha * (
                    reward + gamma * np.max(Q[next_state_idx]) - Q[state_idx, action]
            )

            state_idx = next_state_idx

    return Q

if __name__ == "__main__":
    # Run the Algorithms
    env = CliffWalkingEnv()

    # Run SARSA
    Q_sarsa = sarsa(env)
    print("Q-values from SARSA:")
    print(Q_sarsa)

    # Run Q-Learning
    Q_qlearning = q_learning(env)
    print("Q-values from Q-Learning:")
    print(Q_qlearning)
