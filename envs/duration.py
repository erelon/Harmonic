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
        # --- 1) Snapshot total waiting BEFORE macro-action ---
        start_wait = 0.0
        for sig in self.env.traffic_signals.values():
            # each returns a list of per-lane accumulated waits
            start_wait += sum(sig.get_accumulated_waiting_time_per_lane())
        start_wait /= 100.0  # match env’s scaling

        # --- 2) HOLD loop (ignore rewards) ---
        remaining = {ts: self.min_units + act for ts, act in actions.items()}
        max_chunks = max(remaining.values())
        chunks_executed = 0
        done = {"__all__": False}

        for _ in range(max_chunks):
            # hold current phase
            sub_act = {ts: self.last_phases[ts] for ts in self.env.ts_ids}
            obs, _, done, info = self.env.step(sub_act)
            chunks_executed += 1
            if done["__all__"]:
                break
            for ts in remaining:
                remaining[ts] = max(0, remaining[ts] - 1)

        # --- 3) SWITCH step: flip those with zero remaining ---
        switch_act = {}
        for ts, sig in self.env.traffic_signals.items():
            n_phases = self.env.action_spaces(ts).n
            if remaining[ts] == 0:
                switch_act[ts] = (self.last_phases[ts] + 1) % n_phases
            else:
                switch_act[ts] = self.last_phases[ts]

        obs, _, done, info = self.env.step(switch_act)
        chunks_executed += 1

        # update memory of last phase
        for ts, new_phase in switch_act.items():
            self.last_phases[ts] = new_phase

        # --- 4) Snapshot total waiting AFTER macro-action ---
        end_wait = 0.0
        for sig in self.env.traffic_signals.values():
            end_wait += sum(sig.get_accumulated_waiting_time_per_lane())
        end_wait /= 100.0

        # --- 5) Compute the single “diff-waiting” reward ---
        true_reward = start_wait - end_wait

        # --- 6) Package results ---
        reward_dict = {ts: true_reward for ts in self.env.ts_ids}
        info["action_duration"] = chunks_executed * self.granularity
        return obs, reward_dict, done, info
