import gym
import collections

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
TEST_EPISODES = 20

class Agent:
    
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)

    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            next_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state, action, next_state)] = reward
            self.transits[(self.state, action)][next_state] += 1 
            self.state = self.env.reset() if is_done else next_state
    
    def calculate_action_value(self, state, action):
        target_state_counts = self.transits[(state, action)]
        total_count = sum(target_state_counts.values())
        action_value = 0.0
        for target_state, target_count in target_state_counts.items():
            reward = self.rewards[(state, action, target_state)]
            action_value += (target_count / total_count) * (reward + GAMMA * self.values[target_state])
        return action_value

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            values = [self.calculate_action_value(state, action) 
                      for action in range(self.env.action_space.n)]
            self.values[state] = max(values)

    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calculate_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()

        while True:
            action = self.select_action(state)
            next_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, next_state)] = reward
            self.transits[(state, action)][next_state] += 1 
            total_reward += reward
            if is_done:
                break
            state = next_state
        return total_reward


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent(ENV_NAME)

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break