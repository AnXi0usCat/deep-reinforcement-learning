"""
Agent converts environmental states into actions
"""
import numpy as np
import torch
import torch.nn.functional as F


class BaseAgent:
    """
    Abstract agent interface
    """
    def initial_state(self):
        """
        Should create initial empty state for the agent. It will be called for the start of the episode
        :return: Anything agent want to remember
        """
        return None
    
    def __call__(self, states, agent_states):
        """
        Convert observations and states into actions to take
        :param states: list of environment states to process
        :param agent_states: list of states with the same length as observations
        :return: tuple of actions, states
        """
        assert isinstance(states, list)
        assert isinstance(agent_states, list)
        assert len(agent_states) == len(states)

        raise NotImplementedError


def defalt_state_preprocessor(states):
    """
    Convert list of states into the form suitable for model. By default we assume Variable
    :param states: list of numpy arrays with states
    :return: Variable
    """
    if len(states) == 1:
        states_np = np.array(states)
    else:
        states_np = np.array([np.array(s, copy=False) for s in states], copy=False)
    return torch.tensor(states_np)
