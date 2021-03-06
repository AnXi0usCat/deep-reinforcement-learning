import gym
import collections

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20

class Agent:
    
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.state = self.env.reset()
        self.values = collections.defaultdict(float)

    def sample_env(self):
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, is_done, _ = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state
        return (old_state, action, reward, new_state)

    def value_update(self, s, a, r, next_s):
        best_value, _ = self.best_value_and_action(next_s)
        new_val = r + GAMMA * best_value
        old_val = self.values[(s, a)]
        self.values[(s, a)] = (1-ALPHA) * old_val + ALPHA * new_val
        
    def best_value_and_action(self, state):
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action
    
    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            _, action = self.best_value_and_action(state)
            next_state, reward, is_done, _ = env.step(action)
            total_reward += reward
            if is_done:
                break
            state = next_state
        return total_reward


if __name__ == '__main__':
    test_env = gym.make(ENV_NAME)
    agent = Agent(ENV_NAME)
    
    iter_no = 0
    best_reward = 0.0
    
    while True:
        iter_no += 1
        s, a, r, next_s = agent.sample_env()
        agent.value_update(s, a, r, next_s)
        
        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        
        if reward > best_reward:
            best_reward = reward
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            
        if best_reward >= 0.80:
            print("Solved in %d iterations!" % iter_no)
            break