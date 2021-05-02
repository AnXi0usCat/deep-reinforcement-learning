import gym
import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple


HIDDEN_SIZE = 128
BATCH_SIZE = 100
PERCENTILE = 30
GAMMA = 0.9


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


class DiscreteOneHotWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        super(DiscreteOneHotWrapper, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, (env.observation_space.n, ), dtype=np.float32
        )

    def observation(self, observation):
        res = np.copy(self.observation_space.low)
        res[observation] = 1.0
        return res


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

    # discount the reward depending on how long it took to finish the episode
    disc_rewards = [ep.reward * (GAMMA ** len(ep.steps)) for ep in batch]
    reward_bound = np.percentile(disc_rewards, percentile)

    
    train_states = []
    train_actions = []
    elite_batch = []
    
    for discounted_reward, example in zip(disc_rewards, batch):
        if discounted_reward > reward_bound:
            
            train_states.extend(step.state for step in example.steps)
            train_actions.extend(step.action for step in example.steps)
            elite_batch.append(example)
        
    
    return elite_batch, train_states, train_actions, reward_bound


if __name__ == '__main__':
    
    env = DiscreteOneHotWrapper(gym.make('FrozenLake-v0'))
    n_obs = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    net = Net(n_obs, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001)

    full_batch = []
    for i, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        
        reward_m = float(np.mean(list(map(lambda s: s.reward, batch))))
        full_batch, states, acts, reward_b = filter_batch(batch + full_batch, PERCENTILE)
        
        if not full_batch:
            continue
        
        state_t = torch.FloatTensor(states)
        acts_t = torch.LongTensor(acts)
        full_batch = full_batch[-500:]
        
        optimizer.zero_grad()
        action_scores_t = net(state_t)
        loss = objective(action_scores_t, acts_t)
        
        loss.backward()
        optimizer.step()
        
        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
            i, loss.item(), reward_m, reward_b))
        
        if reward_m > 0.8:
            print("Solved!")
            torch.save(net.state_dict(), 'frozen_lake.pt')
            break