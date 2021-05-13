import numpy as np
import torch
import collections


Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_states'])


class Agent:
    
    def __init__(self, env, buffer):
        self.env = env
        self.buffer = buffer
        self.total_reward = 0.0
        self.state = None
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None
        
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_np = np.array([self.state], copy=False)
            state_t = torch.tensor(state_np).to(device)
            q_vals_t = net(state_t)
            _, action_index = torch.max(q_vals_t, dim=1)
            action = int(action_index.item())
        
        next_state, reward, is_done, _ = self.env.step(action)
        experience = Experience(self.state, action, reward, is_done, next_state)
        self.buffer.append(experience)
        self.total_reward += reward
        self.state = next_state
        
        if is_done:
            done_reward = self.total_reward
            self._reset()
        
        return done_reward
