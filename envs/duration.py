from gym import spaces
from sumo_rl import SumoEnvironment


class DurationWrapper:
    def __init__(
        self,
        net_file,
        route_file,
        min_green,
        max_green,
        granularity=5,
        **kwargs
    ):
        assert min_green % granularity == 0 and max_green % granularity == 0, \
            "min_green and max_green must be multiples of granularity"

        self.granularity = granularity
        self.min_units = min_green // granularity
        self.max_units = max_green // granularity

        self.env = SumoEnvironment(
            net_file=net_file,
            route_file=route_file,
            delta_time=granularity,
            min_green=min_green,
            max_green=max_green,
            **kwargs
        )

        # each ts picks k in [0..max_units-min_units]
        n_units = self.max_units - self.min_units + 1
        self.action_space = spaces.Discrete(n_units)
        self.observation_space = self.env.observation_space

    def reset(self):
        obs = self.env.reset()
        self.last_phases = {
            ts: sig.green_phase for ts, sig in self.env.traffic_signals.items()
        }
        return obs

    def step(self, actions):
        """
        actions: dict ts -> int in [0..max_units-min_units]
        """
        # how many chunks each signal holds its phase
        remaining = {
            ts: self.min_units + act for ts, act in actions.items()
        }
        total_reward = {ts: 0 for ts in self.env.ts_ids}
        done = {"__all__": False}
        info = {}
        chunks_executed = 0

        # A) HOLD loop: run until all remaining counters hit zero
        max_chunks = max(remaining.values())
        for _ in range(max_chunks):
            sub_actions = {ts: self.last_phases[ts] for ts in self.env.ts_ids}
            obs, rew, done, info = self.env.step(sub_actions)
            chunks_executed += 1

            for ts, r in rew.items():
                total_reward[ts] += r
            if done["__all__"]:
                info["action_duration"] = chunks_executed * self.granularity
                return obs, total_reward, done, info

            # decrement counters
            for ts in remaining:
                remaining[ts] = max(0, remaining[ts] - 1)

        # B) SWITCH step: flip those that hit zero
        sub_actions = {}
        for ts, sig in self.env.traffic_signals.items():
            n = self.env.action_spaces(ts).n
            sub_actions[ts] = (
                (self.last_phases[ts] + 1) % n
                if remaining[ts] == 0
                else self.last_phases[ts]
            )

        obs, rew, done, info = self.env.step(sub_actions)
        chunks_executed += 1
        for ts, r in rew.items():
            total_reward[ts] += r

        # update phase memory
        for ts in sub_actions:
            self.last_phases[ts] = sub_actions[ts]

        # total elapsed time in seconds
        info["action_duration"] = chunks_executed * self.granularity
        return obs, total_reward, done, info
