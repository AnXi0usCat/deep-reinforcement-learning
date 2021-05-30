import torch
import torch.nn as nn
import numpy as np
from types import SimpleNamespace
import ptan


SEED = 123

HYPERPARAMS = {
    'pong': SimpleNamespace(**{
        'env_name':         "PongNoFrameskip-v4",
        'stop_reward':      18.0,
        'run_name':         'pong',
        'replay_size':      100000,
        'replay_initial':   10000,
        'target_net_sync':  1000,
        'epsilon_frames':   10**5,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32
    })
}


def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    
    for exp in batch:

        state = np.array(exp.state)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        
        if exp.last_state is None:
            lstate = state
        else:
            lstate = np.array(exp.last_state)
        last_states.append(lstate)
            
    return np.array(states, copy=False), np.array(actions), \ 
           np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), \
           np.array(last_states, copy=False)


def calc_loss_dqn(batch, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = unpack_batch(batch)
    
    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    dones_mask = torch.BoolTensor(dones).to(device)
    
    actions_v = actions_v.unsqueeze(-1)
    state_action_vals = net(states_v).gather(1, actions_v)
    state_action_vals = state_action_vals.squeeze(-1)
    
    with torch.no_grad():
        next_state_vals = tgt_net(next_states_v).max(1)[0]
        next_state_vals[dones_mask] = 0.0
        
    
    expected_action_vals = next_state_vals.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_vals, expected_action_vals)


class EpsilonTracker:
    
    def __init__(self, selector: ptan.actions.EpsilonGreedyActionSelector,
                 params: SimpleNamespace):
        self.selector = selector
        self.params = params
        self.frame(0)

    def frame(self, frame_idx: int):
        eps = self.params.epsilon_start - \
              (frame_idx / self.params.epsilon_frames)
        self.selector.epsilon = max(self.params.epsilon_final, eps)


if __name__ == '__main__':

    random.seed(SEED)
    torch.manual_seed(SEED)
    
    
    