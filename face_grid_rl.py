import numpy as np

class FaceGridEnv:
    def __init__(self, grid_size=(10,10), target_zones=[(3,3), (7,2), (5,8)], max_steps=50):
        self.grid_size = grid_size
        self.target_zones = target_zones
        self.max_steps = max_steps
        self.action_space = 4  # [0=up, 1=down, 2=left, 3=right]
        self.reset()

    def reset(self):
        self.agent_pos = [self.grid_size[0] // 2, self.grid_size[1] // 2]
        self.steps = 0
        self.visited = set()
        return np.array(self.agent_pos, dtype=np.int32)

    def step(self, action):
        if action == 0: self.agent_pos[1] = max(self.agent_pos[1] - 1, 0)
        elif action == 1: self.agent_pos[1] = min(self.agent_pos[1] + 1, self.grid_size[1] - 1)
        elif action == 2: self.agent_pos[0] = max(self.agent_pos[0] - 1, 0)
        elif action == 3: self.agent_pos[0] = min(self.agent_pos[0] + 1, self.grid_size[0] - 1)
        self.steps += 1
        self.visited.add(tuple(self.agent_pos))

        done = self.steps >= self.max_steps
        reward = 0
        if tuple(self.agent_pos) in self.target_zones:
            reward = 10
            done = True
        elif tuple(self.agent_pos) in self.visited:
            reward = -1
        else:
            reward = -0.1
        return np.array(self.agent_pos, dtype=np.int32), reward, done, {}

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.2):
        self.env = env
        self.q_table = np.zeros((*env.grid_size, env.action_space))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(range(self.env.action_space))
        return np.argmax(self.q_table[state[0], state[1], :])

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state[0], state[1], action]
        target = reward + self.gamma * np.max(self.q_table[next_state[0], next_state[1], :])
        self.q_table[state[0], state[1], action] += self.alpha * (target - predict)
