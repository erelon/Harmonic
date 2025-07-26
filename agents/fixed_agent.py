# Baseline Agent: Fixed-time (median action)
class FixedTimeAgent:
    def __init__(self, action_space):
        n = getattr(action_space, 'n', None)
        idx = (n // 2) if n is not None else 0
        self.fixed_action = idx if n is not None else action_space.sample()

    def act(self):
        return self.fixed_action

    def learn(self, *args, **kwargs):
        pass
