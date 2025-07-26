import pickle

from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy

from agents.ql_agent import QLAgent


class RLLearner(QLAgent):
    """R-learning agent class using the rho trick when the best action is taken."""

    def __init__(
            self,
            starting_state,
            state_space,
            action_space,
            alpha=0.1,
            gamma=0.95,  # not used
            rho_learning_rate=0.03,
            exploration_strategy=EpsilonGreedy(),
            q_table_path=None
    ):
        # Initialize base Q-learning parameters
        super().__init__(
            starting_state,
            state_space,
            action_space,
            alpha=alpha,
            gamma=gamma,
            exploration_strategy=exploration_strategy,
            q_table_path=q_table_path
        )
        # Initialize average reward (rho)
        self.rho = 0.0
        self.rho_learning_rate = rho_learning_rate

    def act(self):
        """Choose action based on epsilon-greedy policy inherited from QLAgent."""
        return super().act()

    def learn(self, next_state, reward, done=False):
        """Update Q-table and rho based on R-learning update rule with rho-trick."""
        # Ensure next state is in the Q-table
        if next_state not in self.q_table:
            self.q_table[next_state] = [0 for _ in range(self.action_space.n)]

        s = self.state
        a = self.action
        s1 = next_state

        # Identify greedy actions
        best_current_action = self.q_table[s].index(max(self.q_table[s]))
        best_next_value = max(self.q_table[s1])

        # Compute R-learning temporal difference
        delta = (
                reward
                - self.rho
                + best_next_value
                - self.q_table[s][a]
        )

        # Update Q-value
        self.q_table[s][a] += self.alpha * delta

        # Rho-trick: update rho only if the greedy (best) action was taken
        if a == best_current_action:
            self.rho += self.rho_learning_rate * delta

        # Transition to next state
        self.state = s1
        self.acc_reward += reward

        return self.q_table[s][a], self.rho

    def save_q_table(self, file_path):
        """Save the Q-table and rho to a file."""
        with open(file_path, 'wb') as f:
            pickle.dump({'q_table': self.q_table, 'rho': self.rho}, f)
        print(f"Q-table and rho saved to {file_path}")

    def load_q_table(self, file_path):
        """Load the Q-table and rho from a file."""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.rho = data['rho']
        print(f"Q-table and rho loaded from {file_path}")
