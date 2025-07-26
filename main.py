import argparse
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Ensure SUMO tools are available
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import os

os.environ["LIBSUMO_AS_TRACI"] = "1"

from sumo_rl import SumoEnvironment
from agents import *
from sumo_rl.exploration import EpsilonGreedy
import sumo_rl.nets as nets


def train_and_evaluate(agent_name, agent_builder, args, writer, global_eval_step_ref, base_run_dir):
    training_rewards = []
    eval_avg_rewards = []

    train_steps = int(args.s / max(args.mingreen, 1))
    base_eval_steps = int(args.eval_s / max(args.mingreen, 1))
    eval_steps = 2 * base_eval_steps
    global_train_step = 0

    # Prepare q-tables directory under this run
    qtables_root = os.path.join(base_run_dir, "qtables", agent_name.lower())
    os.makedirs(qtables_root, exist_ok=True)

    for run in range(1, args.runs + 1):
        print(f"\n=== {agent_name} - Starting Training Run {run} ===")
        env = SumoEnvironment(
            net_file=args.net_file,
            route_file=args.route,
            out_csv_name=os.path.join(base_run_dir, "csvs", f"{agent_name}_{run}.csv"),
            use_gui=False,
            num_seconds=args.s,
            min_green=args.mingreen,
            max_green=args.maxgreen,
        )
        initial_states = env.reset()
        agents = {
            ts: agent_builder(
                initial_state=initial_states[ts],
                ts=ts,
                encode_fn=env.encode,
                obs_space=env.observation_space,
                action_space=env.action_space
            )
            for ts in env.ts_ids
        }

        done = {"__all__": False}
        total_train_reward = 0

        with tqdm(total=train_steps, desc=f"{agent_name} Train Run {run}", unit="step") as pbar:
            while not done["__all__"]:
                actions = {ts: agents[ts].act() for ts in agents}
                s, r, done, _ = env.step(actions)
                reward_sum = sum(r.values())
                total_train_reward += reward_sum

                for ts in agents:
                    agents[ts].learn(next_state=env.encode(s[ts], ts), reward=r[ts])

                global_train_step += 1
                writer.add_scalar(f"{agent_name}/StepReward/Train", reward_sum, global_train_step)
                for ts in agents:
                    strat = getattr(agents[ts], "exploration_strategy", None)
                    if strat and hasattr(strat, "epsilon"):
                        writer.add_scalar(f"{agent_name}/Epsilon/{ts}", strat.epsilon, global_train_step)

                pbar.update(1)
                if args.v:
                    pbar.set_postfix(avg_train_reward=total_train_reward / pbar.n)

        # save CSV for this run
        env.save_csv(env.out_csv_name, run)
        training_rewards.append(total_train_reward)
        print(f"=== {agent_name} - Finished Training Run {run} | Reward: {total_train_reward} | Steps: {pbar.n} ===")

        writer.add_scalar(f"{agent_name}/EpisodeReward/Train", total_train_reward, run)

        # zero-out epsilon for evaluation
        original_epsilons = {}
        for ts in agents:
            strat = getattr(agents[ts], "exploration_strategy", None)
            if strat:
                original_epsilons[ts] = (strat.initial_epsilon, strat.min_epsilon)
                strat.initial_epsilon = 0.0
                strat.min_epsilon = 0.0

        env.close()
        eval_env = SumoEnvironment(
            net_file=args.net_file,
            route_file=args.route,
            out_csv_name=None,
            use_gui=False,
            num_seconds=args.eval_s,
            min_green=args.mingreen,
            max_green=args.maxgreen,
        )
        eval_rewards = []
        for ep in range(1, args.eval_episodes + 1):
            print(f"--- {agent_name} Train Run {run} Eval Ep {ep} ---")
            s_eval = eval_env.reset()
            done_eval = {"__all__": False}
            total_eval_reward = 0
            with tqdm(total=eval_steps, desc=f"{agent_name} Eval Run {run} Ep {ep}", unit="step") as eval_pbar:
                while not done_eval["__all__"]:
                    actions = {ts: agents[ts].act() for ts in agents}
                    s_eval, r_eval, done_eval, _ = eval_env.step(actions)
                    step_reward = sum(r_eval.values())
                    total_eval_reward += step_reward

                    global_eval_step_ref[0] += 1
                    writer.add_scalar(f"{agent_name}/StepReward/Eval", step_reward, global_eval_step_ref[0])

                    eval_pbar.update(1)
                    if args.v:
                        eval_pbar.set_postfix(avg_eval_reward=total_eval_reward / eval_pbar.n)

            eval_rewards.append(total_eval_reward)
            print(f"=== {agent_name} Train Run {run} Eval Ep {ep} | Reward: {total_eval_reward} ===")
            writer.add_scalar(
                f"{agent_name}/EpisodeReward/EvalEp",
                total_eval_reward,
                (run - 1) * args.eval_episodes + ep
            )

        avg_eval = sum(eval_rewards) / len(eval_rewards)
        eval_avg_rewards.append(avg_eval)
        writer.add_scalar(f"{agent_name}/EpisodeReward/EvalAvg", avg_eval, run)

        # restore epsilon
        for ts in agents:
            strat = getattr(agents[ts], "exploration_strategy", None)
            if strat and ts in original_epsilons:
                strat.initial_epsilon, strat.min_epsilon = original_epsilons[ts]

        eval_env.close()

        # save q-table per run
        for ts in agents:
            save_path = os.path.join(qtables_root, f"{ts}_run{run}.pkl")
            if hasattr(agents[ts], "save_q_table"):
                agents[ts].save_q_table(save_path)
                print(f"[INFO] Saved Q-table for {ts} to {save_path}")

    return training_rewards, eval_avg_rewards


def evaluate_baseline(agent_name, agent_builder, args, writer, global_eval_step_ref, base_run_dir):
    eval_rewards = []
    base_eval_steps = int(args.eval_s / max(args.mingreen, 1))
    eval_steps = 2 * base_eval_steps

    print(f"\n=== {agent_name} - Baseline Evaluation ===")
    eval_env = SumoEnvironment(
        net_file=args.net_file,
        route_file=args.route,
        out_csv_name=None,
        use_gui=False,
        num_seconds=args.eval_s,
        min_green=args.mingreen,
        max_green=args.maxgreen,
    )
    initial_states = eval_env.reset()
    agents = {
        ts: agent_builder(
            initial_state=initial_states[ts],
            ts=ts,
            encode_fn=eval_env.encode,
            obs_space=eval_env.observation_space,
            action_space=eval_env.action_space
        )
        for ts in eval_env.ts_ids
    }

    for ep in range(1, args.eval_episodes + 1):
        print(f"--- {agent_name} Eval Ep {ep} ---")
        s = eval_env.reset()
        done = {"__all__": False}
        total = 0
        with tqdm(total=eval_steps, desc=f"{agent_name} Eval Ep {ep}", unit="step") as pbar:
            while not done["__all__"]:
                actions = {ts: agents[ts].act() for ts in agents}
                s, r, done, _ = eval_env.step(actions)
                step_reward = sum(r.values())
                total += step_reward

                global_eval_step_ref[0] += 1
                writer.add_scalar(f"{agent_name}/StepReward/Eval", step_reward, global_eval_step_ref[0])

                pbar.update(1)
                if args.v:
                    pbar.set_postfix(avg_eval_reward=total / pbar.n)

        print(f"=== {agent_name} Eval Ep {ep} | Reward: {total} ===")
        writer.add_scalar(f"{agent_name}/EpisodeReward/EvalEp", total, ep)
        eval_rewards.append(total)

    avg_eval = sum(eval_rewards) / len(eval_rewards)
    writer.add_scalar(f"{agent_name}/EpisodeReward/EvalAvg", avg_eval, 1)
    eval_env.close()
    return [], eval_rewards


if __name__ == "__main__":
    sumo_rl_dir = os.path.dirname(nets.__file__)

    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Benchmark Traffic Signal Agents on Single-Intersection SUMO"
    )
    prs.add_argument("-route", type=str,
                     default=f"{sumo_rl_dir}/single-intersection/single-intersection.rou.xml",
                     help="Route definition file")
    prs.add_argument("-net", dest="net_file", type=str,
                     default=f"{sumo_rl_dir}/single-intersection/single-intersection.net.xml",
                     help="Network definition file")
    prs.add_argument("-a", type=float, default=0.1, help="Alpha (learning rate)")
    prs.add_argument("-g", type=float, default=0.99, help="Gamma (discount factor)")
    prs.add_argument("-e", type=float, default=0.1, help="Initial epsilon (exploration)")
    prs.add_argument("-me", type=float, default=0.01, help="Minimum epsilon")
    prs.add_argument("-d", type=float, default=0.99, help="Epsilon decay rate")
    prs.add_argument("-ns", type=int, default=40, help="Fixed green time (NS) for FixedTimeAgent")
    prs.add_argument("-we", type=int, default=40, help="Fixed green time (WE) for FixedTimeAgent")
    prs.add_argument("-mingreen", type=int, default=10, help="Minimum green time")
    prs.add_argument("-maxgreen", type=int, default=50, help="Maximum green time")
    prs.add_argument("-s", type=int, default=100000, help="Training simulation seconds")
    prs.add_argument("-eval_s", type=int, default=20000,
                     help="Evaluation simulation seconds (shorter)")
    prs.add_argument("-v", action="store_true", help="Verbose: show running average reward")
    prs.add_argument("-runs", type=int, default=1, help="Number of training runs for learning agents")
    prs.add_argument("-eval_episodes", type=int, default=1,
                     help="Number of evaluation episodes per agent")
    prs.add_argument("--load_qtable_path", type=str, default=None, help="Path to existing Q-table to load")
    args = prs.parse_args()

    experiment_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_run_dir = os.path.join("runs", experiment_time)

    # create subdirectories
    os.makedirs(os.path.join(base_run_dir, "csvs"), exist_ok=True)
    os.makedirs(os.path.join(base_run_dir, "plots", "single-intersection"), exist_ok=True)

    writer = SummaryWriter(log_dir=base_run_dir)
    global_eval_step_ref = [0]

    agent_builders = {
        "Q-Learning": lambda initial_state, ts, encode_fn, obs_space, action_space: QLAgent(
            starting_state=encode_fn(initial_state, ts),
            state_space=obs_space,
            action_space=action_space,
            alpha=args.a,
            gamma=args.g,
            q_table_path=args.load_qtable_path,
            exploration_strategy=EpsilonGreedy(
                initial_epsilon=args.e,
                min_epsilon=args.me,
                decay=args.d
            ),
        ),
        "R-Learning": lambda initial_state, ts, encode_fn, obs_space, action_space: RLLearner(
            starting_state=encode_fn(initial_state, ts),
            state_space=obs_space,
            action_space=action_space,
            alpha=args.a,
            gamma=args.g,
            q_table_path=args.load_qtable_path,
            rho_learning_rate=args.a,
            exploration_strategy=EpsilonGreedy(
                initial_epsilon=args.e,
                min_epsilon=args.me,
                decay=args.d
            ),
        ),
        "Random": lambda initial_state, ts, encode_fn, obs_space, action_space: RandomAgent(action_space),
        "FixedTime": lambda initial_state, ts, encode_fn, obs_space, action_space: FixedTimeAgent(action_space),
    }

    results = {}
    baseline_agents = {"Random", "FixedTime"}
    for name, builder in agent_builders.items():
        global_eval_step_ref[0] = 0

        if name in baseline_agents:
            train_rewards, eval_rewards = evaluate_baseline(
                name, builder, args, writer, global_eval_step_ref, base_run_dir
            )
        else:
            train_rewards, eval_rewards = train_and_evaluate(
                name, builder, args, writer, global_eval_step_ref, base_run_dir
            )

        results[name] = {"train": train_rewards, "eval": eval_rewards}

    runs = list(range(1, args.runs + 1))
    plt.figure(figsize=(10, 6))
    for name, res in results.items():
        # plot training series (always one point per run)
        if res["train"]:
            x_train = list(range(1, len(res["train"]) + 1))
            plt.plot(x_train, res["train"], marker='o', label=f"{name} Train")

        # plot evaluation series (may be per‐run or per‐episode)
        x_eval = list(range(1, len(res["eval"]) + 1))
        plt.plot(x_eval, res["eval"], marker='x', linestyle='--', label=f"{name} Eval")

    plt.title("Training vs Evaluation Rewards")
    plt.xlabel("Run / Eval Ep")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # save plot under runs directory
    plot_path = os.path.join(
        base_run_dir, "plots", "single-intersection",
        f"{experiment_time}_performance_plot.png"
    )
    plt.savefig(plot_path)
    print(f"Saved performance plot to {plot_path}")

    writer.close()
