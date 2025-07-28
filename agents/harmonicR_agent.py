from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy
from . import RLLearner


class HarmonicRLAgent(RLLearner):
    """
    Continuous reinforcement learner using Schwartz's R-learning harmonic update,
    maintaining an exponential moving average of reciprocal rewards for rho.
    """

    def __init__(
            self,
            starting_state,
            state_space,
            action_space,
            alpha=0.1,
            gamma=0.95,
            rho_learning_rate=0.3,
            exploration_strategy=EpsilonGreedy(),
            q_table_path=None
    ):
        super().__init__(
            starting_state,
            state_space,
            action_space,
            alpha=alpha,
            gamma=gamma,
            rho_learning_rate=rho_learning_rate,
            exploration_strategy=exploration_strategy,
            q_table_path=q_table_path
        )
        # Initialize harmonic averaging for rho
        self.reciprocal_rho = 1.0

    def reset(self):
        """Reset the harmonic rho trackers."""
        self.reciprocal_rho = 1.0

    def learn(self, next_state, reward, done=False, time=None):
        """
        Update the agent's Q-table and average reward (rho) using harmonic updates.
        Only updates rho on greedy actions via EMA of reciprocal intervals.
        """
        s = self.state
        a = self.action

        """Update Q-table with new experience."""
        if next_state not in self.q_table:
            self.q_table[next_state] = [0 for _ in range(self.action_space.n)]

        best_next_action = self.q_table[next_state].index(max(self.q_table[next_state]))
        best_current_action = self.q_table[s].index(max(self.q_table[s]))

        delta = (
                reward
                - self.rho * time
                + self.q_table[next_state][best_next_action]
                - self.q_table[s][a]
        )

        # Q-value update
        self.q_table[s][a] += self.rho_learning_rate * delta

        # Harmonic rho update on greedy actions
        if a == best_current_action:
            # EMA of reciprocal times per reward
            self.reciprocal_rho = (
                    (1 - self.rho_learning_rate) * self.reciprocal_rho
                    + self.rho_learning_rate *
                    (time / (reward + 1e-8))  # reward can be 0
            )
            self.rho = 1.0 / self.reciprocal_rho

        self.state = next_state
        self.acc_reward += reward

        return self.q_table[s][a], self.rho
