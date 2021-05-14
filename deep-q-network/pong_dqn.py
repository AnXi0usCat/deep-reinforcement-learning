import torch.nn as nn

from lib import agent
from lib import buffer
from lib import model
from lib import wrapppers


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.5

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02


def calc_loss(batch, net, tgt_net, device="cpu", discount_factor=GAMMA):
    
    states, actions, rewards, dones, next_states = batch
    
    states_t = torch.tensor(states).to(device)
    actions_t = torch.tensor(actions).to(device)
    rewards_t = torch.tensor(rewards).to(device)           
    next_states_t = torch.tensor(next_states).to(device)    
    dones_t = torch.ByteTensor(dones_t).to(device)
    
    state_action_values = net(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
    next_state_action_values = tgt_net(next_states_t).max(dim=1)[0]
    next_state_action_values[dones_t] = 0.0
    next_state_action_values = next_state_action_values.detach()
    
    expected_state_action_values = rewards_t + discount_factor * next_state_action_values
    
    return nn.MSELoss(state_action_values, expected_state_action_values)
