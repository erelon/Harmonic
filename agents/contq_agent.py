import math

import numpy as np
from sumo_rl.exploration import EpsilonGreedy

from . import QLAgent

# Maximum reward placeholder for initializing continuous action Q-values
MAX_REWARDS = float('inf')


class ContinuousQLearningAgent(QLAgent):
    """
    Continuous Q-learning agent that learns from the environment.
    This agent is designed for environments with continuous action spaces.
    """

    def __init__(self, starting_state, state_space, action_space, alpha=0.1, gamma=0.95,
                 exploration_strategy=EpsilonGreedy(), q_table_path=None, _lambda=0.01):
        super().__init__(starting_state, state_space, action_space, alpha, gamma, exploration_strategy, q_table_path)
        self._lambda = _lambda

    def learn(self, next_state, reward, done=False, time=None):
        """
        Update the Q-value based on the action taken and the reward received.
        This method is adapted for continuous action spaces.
        """
        s = self.state
        a = self.action

        """Update Q-table with new experience."""
        if next_state not in self.q_table:
            self.q_table[next_state] = [0 for _ in range(self.action_space.n)]

        # Select best next action
        best_next_action = np.argmax(self.q_table[next_state])

        # Continuous-time discount factor adjustment
        df = math.exp(-self._lambda * time * self.gamma)

        # Temporal-difference target and error
        td_target = reward + df * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[s][a]

        # Update Q-value
        self.q_table[s][a] += self.alpha * td_error

        self.state = next_state
        self.acc_reward += reward
