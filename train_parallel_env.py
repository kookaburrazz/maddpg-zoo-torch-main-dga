"""
Training script for MADDPG with PettingZoo using parallel environments
- Works on Linux + Windows
- Fixes Supersuit vec-env issues that cause "Killed" on simple_spread_v3
- Avoids emoji/Unicode print crashes on Windows consoles
"""

import os
import argparse
import random
import platform
import time
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

import supersuit as ss
from pettingzoo.mpe import simple_tag_v3, simple_spread_v3, simple_adversary_v3

from maddpg import MADDPG, ReplayBuffer
from utils.env import get_env_info, ENV_MAP
from utils.logger import Logger
from utils.utils import evaluate


# -------------------------
# Helpers
# -------------------------
def safe_print(msg: str):
    """
    Avoid Windows cp1252 UnicodeEncodeError and random console encoding issues.
    """
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore"))


def build_env_fn(env_name: str, max_steps: int):
    ENV_MODULES = {
        "simple_tag_v3": simple_tag_v3,
        "simple_spread_v3": simple_spread_v3,
        "simple_adversary_v3": simple_adversary_v3,
    }
    if env_name not in ENV_MODULES:
        raise ValueError(f"Environment {env_name} not manually supported yet.")

    env_module = ENV_MODULES[env_name]

    def env_fn():
        env = env_module.parallel_env(max_cycles=max_steps, continuous_actions=True, render_mode=None)
        env = ss.pad_observations_v0(env)
        env = ss.pad_action_space_v0(env)
        return env

    return env_fn


def create_parallel_env_auto(env_name: str, max_steps: int, num_envs: int):
    """
    IMPORTANT FIX:
      - simple_spread_v3 sometimes crashes / gets SIGKILL when using supersuit concat_vec_envs
        depending on supersuit/gymnasium versions and multiprocessing settings.
      - For stability, we disable concat for simple_spread_v3 and run a single vec env.
        (Baseline and DAG both use the same env pipeline => fair comparison.)
    """

    env_fn = build_env_fn(env_name, max_steps)

    system_name = platform.system()

    # Always create a base vec env (single)
    base_vec = ss.pettingzoo_env_to_vec_env_v1(env_fn())

    # ---- Special-case: simple_spread_v3 stable mode (NO concat) ----
    if env_name == "simple_spread_v3":
        safe_print("simple_spread_v3: using SINGLE vec env (no concat_vec_envs) for stability.")
        return base_vec

    # ---- Other envs: try concat_vec_envs for speed ----
    if system_name == "Windows":
        # Multiprocessing can be flaky on Windows; use serial
        safe_print("Windows detected: using serial vec env (num_cpus=0) for stability.")
        n_cpus = 0
    else:
        safe_print(f"Linux/Unix detected: using concat_vec_envs (num_cpus={num_envs}).")
        n_cpus = max(0, int(num_envs))

    # Prefer gymnasium base_class
    try:
        vec_env = ss.concat_vec_envs_v1(base_vec, num_envs, num_cpus=n_cpus, base_class="gymnasium")
        return vec_env
    except Exception as e:
        safe_print(f"[WARN] concat_vec_envs_v1 failed with gymnasium: {e}")
        safe_print("[WARN] Falling back to serial concat (num_cpus=0, base_class='gymnasium').")
        try:
            vec_env = ss.concat_vec_envs_v1(base_vec, num_envs, num_cpus=0, base_class="gymnasium")
            return vec_env
        except Exception as e2:
            safe_print(f"[WARN] serial concat_vec_envs_v1 still failed: {e2}")
            safe_print("[WARN] Falling back to SINGLE vec env (no concat).")
            return base_vec


# -------------------------
# Args
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-name",
        type=str,
        default="simple_adversary_v3",
        choices=list(ENV_MAP.keys()),
        help="Name of the environment to use",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Run name")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--total-steps", type=int, default=int(1e6), help="Number of steps")
    parser.add_argument("--buffer-size", type=int, default=int(1e6), help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--max-steps", type=int, default=25, help="Maximum steps per episode")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="Soft update parameter")
    parser.add_argument("--actor-lr", type=float, default=1e-3, help="Actor learning rate")
    parser.add_argument("--critic-lr", type=float, default=1e-3, help="Critic learning rate")
    parser.add_argument("--hidden-sizes", type=str, default="256,256", help="Hidden layer sizes (comma-separated)")
    parser.add_argument("--update-every", type=int, default=1, help="Update networks every n steps")
    parser.add_argument("--num-updates", type=int, default=4, help="Number of updates per step")
    parser.add_argument("--noise-scale", type=float, default=0.3, help="Initial Gaussian noise scale")
    parser.add_argument("--min-noise", type=float, default=0.05, help="Minimum Gaussian noise scale")
    parser.add_argument("--noise-decay-steps", type=int, default=int(5e5), help="Decay steps")
    parser.add_argument("--use-noise-decay", action="store_true", default=True, help="Use noise decay")
    parser.add_argument("--eval-interval", type=int, default=5000, help="Evaluate every n steps")
    parser.add_argument("--render-mode", type=str, default=None, choices=[None, "human", "rgb_array"])
    parser.add_argument("--create-gif", action="store_true", help="Create GIF of episodes")

    # DAG switches
    parser.add_argument("--use-dag", action="store_true", default=False, help="Use DAG-Attention Critic")
    parser.add_argument("--dag-no-gate", action="store_true", help="Ablate gating (fixed 0.5)")
    parser.add_argument("--dag-no-decouple", action="store_true", help="Ablate decoupling (shared attention)")

    # Seed
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")

    return parser.parse_args()


# -------------------------
# Train
# -------------------------
def train(args):
    # Lock random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    safe_print(f"[Seed] Random Seed set to: {args.seed}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build dag tag for run naming
    if args.use_dag:
        dag_tag = "DAG"
        if args.dag_no_gate:
            dag_tag += "_NoGate"
        if args.dag_no_decouple:
            dag_tag += "_NoDecouple"
    else:
        dag_tag = "MLP"

    if args.run_name:
        experiment_name = f"{args.run_name}_{dag_tag}_seed{args.seed}_{timestamp}"
    else:
        experiment_name = f"parallel_{dag_tag}_seed{args.seed}_envs{args.num_envs}_{timestamp}"

    logger = Logger(run_name=experiment_name, folder="runs", algo="MADDPG", env=args.env_name)
    logger.log_all_hyperparameters(vars(args))

    # env info
    agents, num_agents, action_sizes, action_low, action_high, state_sizes = get_env_info(
        env_name=args.env_name, max_steps=args.max_steps, apply_padding=True
    )

    # grouping
    if "spread" in args.env_name:
        agent_groups = [list(range(num_agents))]
    elif "tag" in args.env_name:
        adversaries = list(range(num_agents - 1))
        good_agents = [num_agents - 1]
        agent_groups = [adversaries, good_agents]
    else:
        agent_groups = [list(range(num_agents))]

    safe_print(f"Agent Groups: {agent_groups}")

    # parallel env (auto)
    num_envs_req = max(1, int(args.num_envs))
    env = create_parallel_env_auto(args.env_name, args.max_steps, num_envs_req)

    # IMPORTANT:
    # If create_parallel_env_auto returned a SINGLE vec env for simple_spread_v3,
    # then the real num_envs is 1 for shaping / stepping.
    real_num_envs = num_envs_req
    try:
        # Gymnasium VecEnv typically has num_envs attribute
        if hasattr(env, "num_envs"):
            real_num_envs = int(env.num_envs)
    except Exception:
        pass

    # eval env
    from utils.env import create_single_env

    env_evaluate = create_single_env(args.env_name, args.max_steps, "rgb_array", True)

    model_path = os.path.join(logger.dir_name, "model.pt")
    best_model_path = os.path.join(logger.dir_name, "best_model.pt")
    best_score = -float("inf")

    hidden_sizes = tuple(map(int, args.hidden_sizes.split(",")))

    # Model
    maddpg = MADDPG(
        state_sizes=state_sizes,
        action_sizes=action_sizes,
        hidden_sizes=hidden_sizes,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
        action_low=action_low,
        action_high=action_high,
        agent_groups=agent_groups,
        use_dag_attention=args.use_dag,
        dag_no_gate=args.dag_no_gate,
        dag_no_decouple=args.dag_no_decouple,
    )

    # Replay buffer
    buffer = ReplayBuffer(
        buffer_size=min(args.buffer_size, args.total_steps),
        batch_size=args.batch_size,
        agents=agents,
        state_sizes=state_sizes,
        action_sizes=action_sizes,
    )

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    safe_print(f"[Train] Starting Training on {platform.system()} | GPU: {gpu_name}")
    safe_print(f"   Mode: {dag_tag} | Envs(requested): {num_envs_req} | Envs(actual): {real_num_envs}")
    safe_print(f"   Batch: {args.batch_size} | Seed: {args.seed}")
    safe_print(f"   DAG flags: no_gate={args.dag_no_gate}, no_decouple={args.dag_no_decouple}")

    noise_scale = float(args.noise_scale)
    # decay steps should respect actual env count
    decay_steps = min(args.noise_decay_steps // max(1, real_num_envs), args.total_steps // max(1, real_num_envs))
    decay_steps = max(1, int(decay_steps))
    noise_decay = (args.noise_scale - args.min_noise) / decay_steps

    agent_rewards = [[] for _ in range(len(agents))]
    global_step = 0
    # eval_interval in terms of env-steps (global_step counts env transitions)
    eval_interval = max(1, (args.eval_interval // max(1, real_num_envs)) * max(1, real_num_envs))

    # Initial eval
    evaluate(env_evaluate, maddpg, logger, record_gif=False, num_eval_episodes=5, global_step=0)

    with tqdm(total=args.total_steps, desc=f"Training (Seed {args.seed})") as pbar:
        while global_step < args.total_steps:
            observations, _ = env.reset()
            observations_reshaped = observations.reshape(real_num_envs, num_agents, -1)
            episode_rewards = np.zeros((real_num_envs, num_agents), dtype=np.float32)

            for _step in range(args.max_steps):
                states_batched = [observations_reshaped[:, i, :] for i in range(num_agents)]
                actions_batched = maddpg.act(states_batched, add_noise=True, noise_scale=noise_scale)

                actions_stacked = np.stack(actions_batched, axis=1)  # [E, N, A]
                actions_array = actions_stacked.reshape(real_num_envs * num_agents, -1)

                next_obs, rewards, terminations, truncations, infos = env.step(actions_array)
                dones = np.logical_or(terminations, truncations)

                next_obs_reshaped = next_obs.reshape(real_num_envs, num_agents, -1)
                rewards_reshaped = rewards.reshape(real_num_envs, num_agents)
                terminations_reshaped = terminations.reshape(real_num_envs, num_agents)

                # store transitions
                for env_idx in range(real_num_envs):
                    buffer.add(
                        states=observations_reshaped[env_idx],
                        actions=actions_stacked[env_idx],
                        rewards=rewards_reshaped[env_idx],
                        next_states=next_obs_reshaped[env_idx],
                        dones=terminations_reshaped[env_idx],
                    )

                episode_rewards += rewards_reshaped
                observations_reshaped = next_obs_reshaped

                # learn
                if len(buffer) > args.batch_size and global_step % args.update_every == 0:
                    for _u in range(args.num_updates):
                        experiences = buffer.sample()
                        for i in range(num_agents):
                            critic_loss, actor_loss = maddpg.learn(experiences, i)
                            if global_step % 100 == 0:
                                logger.add_scalar(f"{agents[i]}/critic_loss", critic_loss, global_step)
                                logger.add_scalar(f"{agents[i]}/actor_loss", actor_loss, global_step)
                        maddpg.update_targets()

                if args.use_noise_decay:
                    noise_scale = max(args.min_noise, noise_scale - noise_decay)

                global_step += real_num_envs
                pbar.update(real_num_envs)

                if np.any(dones):
                    break

            # log episode rewards
            for agent_idx in range(num_agents):
                mean_r = float(np.mean(episode_rewards[:, agent_idx]))
                agent_rewards[agent_idx].append(
                    [mean_r, float(np.min(episode_rewards[:, agent_idx])), float(np.max(episode_rewards[:, agent_idx]))]
                )
                logger.add_scalar(f"{agents[agent_idx]}/episode_reward", mean_r, global_step)

            logger.add_scalar("noise/scale", noise_scale, global_step)
            total_avg_reward = float(np.sum(episode_rewards) / max(1, real_num_envs))
            logger.add_scalar("train/total_reward", total_avg_reward, global_step)

            # evaluate
            if global_step % eval_interval == 0:
                maddpg.save(model_path)
                create_gif = args.create_gif and (global_step % (eval_interval * 4) == 0)

                avg_eval_rewards = evaluate(
                    env_evaluate,
                    maddpg,
                    logger,
                    num_eval_episodes=5,
                    record_gif=create_gif,
                    global_step=global_step,
                )

                score = float(np.sum(avg_eval_rewards))
                if score > best_score:
                    best_score = score
                    maddpg.save(best_model_path)
                    safe_print(f"[Best] New Best: {best_score:.2f}")

    maddpg.save(model_path)
    env.close()
    env_evaluate.close()
    logger.close()

    save_file_name = f"agent_rewards_seed{args.seed}.npy"
    save_path = os.path.join(logger.dir_name, save_file_name)
    np.save(save_path, agent_rewards)
    safe_print(f"[Save] Results saved to: {save_path}")

    # also dump a copy to cwd
    np.save(save_file_name, agent_rewards)
    safe_print(f"[Save] Also saved copy to current folder: {save_file_name}")

    return agent_rewards, experiment_name


if __name__ == "__main__":
    args = parse_args()
    train(args)
