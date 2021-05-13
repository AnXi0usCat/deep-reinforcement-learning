import collections
import numpy as np

class ReplayBuffer:
    
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    
    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, new_states = zip(*[self.buffer[index] for index in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(new_states)
