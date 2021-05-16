import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from lib import agent
from lib import buffer
from lib import model
from lib import wrappers


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
    dones_t = torch.ByteTensor(dones).to(device)
    
    state_action_values = net(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
    next_state_action_values = tgt_net(next_states_t).max(dim=1)[0]
    next_state_action_values[dones_t] = 0.0
    next_state_action_values = next_state_action_values.detach()
    
    expected_state_action_values = rewards_t + discount_factor * next_state_action_values
    
    return nn.MSELoss(state_action_values, expected_state_action_values)


if __name__ == '__main__':
    
    # select the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create the new environment
    env = wrappers.make_env(DEFAULT_ENV_NAME)
    
    # create the base end target networks
    net = model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    
    # intialise the replay buffer and the agent
    buffer = buffer.ReplayBuffer(REPLAY_SIZE)
    agent = agent.Agent(env, buffer)
    epsilon = EPSILON_START
    
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    best_mean_reward = None
    
    while True:
        frame_idx += 1
        # slect the max of either epsilon final or decayed epsilon start
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
        
        # play one step
        reward = agent.play_step(net, epsilon, device)
        if reward is not None:
            total_rewards.append(reward)
            mean_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, mean reward %.3f, eps %.2f f/s" % (
                frame_idx, len(total_rewards), mean_reward, epsilon))
            
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-best.dat")
                best_mean_reward = mean_reward
                print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
            
            if best_mean_reward >= MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break
            
        
        # if the buffer is too small, restart the loop
        if len(buffer) < REPLAY_START_SIZE:
            continue
        
        # if 1000 episodes passed, sync the networks
        if frame_idx % SYNC_TARGET_FRAMES:
            tgt_net.load_state_dict(net.state_dict())
        
        # backpropogate the change in the base network
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss = calc_loss(batch, net, tgt_net, device, GAMMA)
        loss.backward()
        optimizer.step()
        
    
    
    
    
    
    
    
    