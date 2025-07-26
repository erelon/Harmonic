import pickle

from sumo_rl.agents import QLAgent as QLAgentBase
from sumo_rl.exploration import EpsilonGreedy


class QLAgent(QLAgentBase):
    def __init__(self, starting_state, state_space, action_space, alpha=0.5, gamma=0.95,
                 exploration_strategy=EpsilonGreedy(), q_table_path=None):
        """Initialize Q-learning agent."""
        super().__init__(starting_state, state_space, action_space, alpha, gamma, exploration_strategy)
        if q_table_path:
            self.load_q_table(q_table_path)

    def save_q_table(self, file_path):
        """Save the Q-table and rho to a file."""
        with open(file_path, 'wb') as f:
            pickle.dump({'q_table': self.q_table}, f)
        print(f"Q-table saved to {file_path}")

    def load_q_table(self, file_path):
        """Load the Q-table and rho from a file."""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
        print(f"Q-table loaded from {file_path}")
