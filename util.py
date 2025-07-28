from agents import *
from sumo_rl.exploration import EpsilonGreedy


def build_q_learning(init_state, ts, encode_fn, obs_space, act_space, args):
    return QLAgent(
        starting_state=encode_fn(init_state, ts),
        state_space=obs_space,
        action_space=act_space,
        alpha=args.a, gamma=args.g,
        q_table_path=args.load_qtable_path,
        exploration_strategy=EpsilonGreedy(args.e, args.me, args.d)
    )


def build_continuous_q(init_state, ts, encode_fn, obs_space, act_space, args):
    return ContinuousQLearningAgent(
        starting_state=encode_fn(init_state, ts),
        state_space=obs_space, action_space=act_space,
        alpha=args.a, gamma=args.g,
        q_table_path=args.load_qtable_path,
        exploration_strategy=EpsilonGreedy(args.e, args.me, args.d)
    )


def build_r_learning(init_state, ts, encode_fn, obs_space, act_space, args):
    return RLLearner(
        starting_state=encode_fn(init_state, ts),
        state_space=obs_space, action_space=act_space,
        alpha=args.a, gamma=args.g,
        q_table_path=args.load_qtable_path,
        rho_learning_rate=args.a,
        exploration_strategy=EpsilonGreedy(args.e, args.me, args.d)
    )


def build_smart(init_state, ts, encode_fn, obs_space, act_space, args):
    return SMARTRLAgent(
        starting_state=encode_fn(init_state, ts),
        state_space=obs_space, action_space=act_space,
        alpha=args.a, gamma=args.g,
        q_table_path=args.load_qtable_path,
        rho_learning_rate=args.a,
        exploration_strategy=EpsilonGreedy(args.e, args.me, args.d)
    )


def build_harmonic(init_state, ts, encode_fn, obs_space, act_space, args):
    return HarmonicRLAgent(
        starting_state=encode_fn(init_state, ts),
        state_space=obs_space, action_space=act_space,
        alpha=args.a, gamma=args.g,
        q_table_path=args.load_qtable_path,
        rho_learning_rate=args.a,
        exploration_strategy=EpsilonGreedy(args.e, args.me, args.d)
    )


def build_random(init_state, ts, encode_fn, obs_space, act_space, args):
    return RandomAgent(act_space)


def build_fixed_time(init_state, ts, encode_fn, obs_space, act_space, args):
    return FixedTimeAgent(act_space)


# 2) Module‐level mapping from name → builder function
AGENT_BUILDERS = {
    "Q-Learning": build_q_learning,
    "Continuous-Q-Learning": build_continuous_q,
    "R-Learning": build_r_learning,
    "SMART": build_smart,
    "Harmonic-R-Learning": build_harmonic,
    "Random": build_random,
    "FixedTime": build_fixed_time,
}

BASELINE_AGENTS = {"Random", "FixedTime"}
