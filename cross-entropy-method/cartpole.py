import gym
import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple


HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['state', 'action'])


class Net(nn.Module):

    def __init__(self, n_obs, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


def iterate_batches(env, net, batch_size):
    
    batch = []
    episode_reward = 0.0
    episode_steps = []
    state = env.reset()
    soft_max = nn.Softmax(dim=1) 
    
    while True:

        state_t = torch.FloatTensor([state])
        actions_prob_t = soft_max(net(state_t))
        actions_prob = actions_prob_t.data.numpy()[0]
        
        action = np.random.choice(len(actions_prob), p=actions_prob)
        next_state, reward, is_done, _ = env.step(action)
        
        episode_steps.append(EpisodeStep(state, action))
        episode_reward += reward

        if is_done:
            batch.append(Episode(episode_reward, episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            
            if len(batch) == batch_size:
                yield batch
                batch = []

        state = next_state


def filter_batch(batch, percentile):

    rewards = [ep.reward for ep in batch]
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))
    
    train_states = []
    train_actions = []
    
    for example in batch:
        if example.reward < reward_bound:
            continue
            
        train_states.extend(step.state for step in example.steps)
        train_actions.extend(step.action for step in example.steps)
        
        train_states_t = torch.FloatTensor(train_states)
        train_actions_t = torch.LongTensor(train_actions)
    
    return train_states_t, train_actions_t, reward_bound, reward_mean


if __name__ == '__main__':
    
    env = gym.make('CartPole-v0')
    n_obs = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    net = Net(n_obs, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.01)

    for i, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        state_t, acts_t, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        
        optimizer.zero_grad()
        action_scores_t = net(state_t)
        loss = objective(action_scores_t, acts_t)
        
        loss.backward()
        optimizer.step()
        
        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
            i, loss.item(), reward_m, reward_b))
        
        if reward_m > 199:
            print("Solved!")
            torch.save(net.state_dict(), 'cartpole.pt')
            break