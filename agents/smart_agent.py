from sumo_rl.exploration import EpsilonGreedy

from agents import RLLearner


class SMARTRLAgent(RLLearner):
    """
    Continuous reinforcement learner based on Schwartz's R-learning algorithm,
    using continuous rewards and time-based rho updates.
    """

    def __init__(
            self,
            starting_state,
            state_space,
            action_space,
            alpha=0.1,
            gamma=0.95,
            rho_learning_rate=0.03,
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
        # Initialize timing and reward accumulators for rho update
        self.total_time = 0
        self.total_reward = 0

    def learn(self, next_state, reward, done=False, time=None):
        """
        Update the agent's Q-table and average reward (rho) based on the last action,
        reward, and elapsed time. Always updates rho on greedy actions.
        """
        s = self.state
        a = self.action

        """Update Q-table with new experience."""
        if next_state not in self.q_table:
            self.q_table[next_state] = [0 for _ in range(self.action_space.n)]

        # Compute optimal actions
        best_next_action = self.q_table[next_state].index(max(self.q_table[next_state]))
        best_current_action = self.q_table[s].index(max(self.q_table[s]))

        # Temporal-difference calculation
        delta = (
                reward
                - self.rho * time
                + self.q_table[next_state][best_next_action]
                - self.q_table[s][a]
        )

        # Q-value update
        self.q_table[s][a] += self.rho_learning_rate * delta

        # Rho update on greedy action
        if a == best_current_action:
            self.total_time += time
            self.total_reward += reward
            self.rho = self.total_reward / self.total_time

        # Transition and accumulate
        self.state = next_state
        self.acc_reward += reward

        return self.q_table[s][a], self.rho
