import numpy as np
import time
import abc
from typing import Union


class ActionSelector(abc.ABC):
    """
    Abstract class which converts scores to the actions
    """

    @abc.abstractmethod
    def __call__(self, score):
        pass


class ArgmaxActionSelector(ActionSelector):
    """
    Selects actions using argmax
    """

    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        return np.argmax(scores, dim=1)


class EpsilonGreedyActionSelector(ActionSelector):
    """
    Selects actions using Epsilon greedy policy
    """
    
    def __init__(self, epsilon=0.05, selecor=None):
        self.epsilon = epsilon
        self.selector = selector if selector is not None else ArgmaxActionSelector()

    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        batch_size, n_actions = scores.shape
        actions = self.selector(scores)
        mask = np.random.random(size=batch_size) < epsilon
        random_actions = np.random.choice(n_actions, sum(mask))
        actions[mask] = random_actions
        return actions


class ProbabilityActionSelector(ActionSelector):
    """
    Converts probabilities of actions into action by sampling them
    """ 

    def __call__(self, probs):
        assert isinstance(scores, np.ndarray)
        actions = []
        for prob in probs:
            actions.append(np.random.choice(len(prob), p=prob))
        return np.array(actions)


class EpsilonTracker:
    """
    Updates epsilon according to linear schedule
    """
    
    def __init__(self, selector: EpsilonGreedyActionSelector,                  
                 eps_start: Union[int, float],
                 eps_final: Union[int, float],
                 eps_frames: int):
        self.selector = selector
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_frames = eps_frames
        self.frame(0)

    def frame(self, frame: int):
        eps = self.eps_start - frame / self.eps_frame
        self.selector.epsilon = max(self.eps_final, eps)


class RewardTracker:
    """
    A Context manager that prints out the mean reward received
    by the agent over a period of time
    """
    
    def __init__(self, stop_reward):
        self.stop_reward = stop_reward

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        pass

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        epslilon_str = '' if epsilon is None else ', eps %.2f' % epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards), mean_reward, speed, epsilon_str
        ))
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False