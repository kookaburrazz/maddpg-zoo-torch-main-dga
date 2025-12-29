import warnings
import logging
import gymnasium as gym

# 1. 屏蔽烦人的警告
warnings.filterwarnings("ignore")

import torch
import numpy as np
import argparse
import os
import time
from datetime import datetime
from collections import deque

from maddpg.maddpg import MADDPG
from utils.env import create_single_env, ENV_MAP, get_env_info

# 自动检测设备 (有显卡用显卡)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training using device: {device}")


def parse_args():
    parser = argparse.ArgumentParser("MADDPG Training with DAG Attention")
    parser.add_argument("--env-name", type=str, default="simple_spread_v3", choices=list(ENV_MAP.keys()))
    parser.add_argument("--max-steps", type=int, default=25, help="Max steps per episode")
    parser.add_argument("--episodes", type=int, default=20000, help="Total training episodes")  # 增加到2万轮
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr-actor", type=float, default=1e-4)
    parser.add_argument("--lr-critic", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--use-dag", action="store_true", default=False, help="Enable DAG-Attention Critic")
    parser.add_argument("--save-dir", type=str, default="./runs", help="Directory to save models")
    parser.add_argument("--save-rate", type=int, default=1000, help="Save model every N episodes")
    parser.add_argument("--print-rate", type=int, default=50, help="Print progress every N episodes")  # 默认改为50
    parser.add_argument("--run-name", type=str, default=None, help="Name of the run for output directory")
    return parser.parse_args()


class ReplayBuffer:
    def __init__(self, capacity, num_agents, state_sizes, action_sizes):
        self.capacity = capacity
        self.num_agents = num_agents
        self.ptr = 0
        self.size = 0
        self.states = [np.zeros((capacity, s)) for s in state_sizes]
        self.actions = [np.zeros((capacity, a)) for a in action_sizes]
        self.rewards = [np.zeros((capacity, 1)) for _ in range(num_agents)]
        self.next_states = [np.zeros((capacity, s)) for s in state_sizes]
        self.dones = [np.zeros((capacity, 1)) for _ in range(num_agents)]

    def add(self, states, actions, rewards, next_states, dones):
        for i in range(self.num_agents):
            self.states[i][self.ptr] = states[i]
            self.actions[i][self.ptr] = actions[i]
            self.rewards[i][self.ptr] = rewards[i]
            self.next_states[i][self.ptr] = next_states[i]
            self.dones[i][self.ptr] = dones[i]
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.choice(self.size, batch_size)
        states = [torch.tensor(self.states[i][idx], dtype=torch.float32).to(device) for i in range(self.num_agents)]
        actions = [torch.tensor(self.actions[i][idx], dtype=torch.float32).to(device) for i in range(self.num_agents)]
        rewards = [torch.tensor(self.rewards[i][idx], dtype=torch.float32).to(device) for i in range(self.num_agents)]
        next_states = [torch.tensor(self.next_states[i][idx], dtype=torch.float32).to(device) for i in
                       range(self.num_agents)]
        dones = [torch.tensor(self.dones[i][idx], dtype=torch.float32).to(device) for i in range(self.num_agents)]

        states_full = torch.cat(states, dim=1)
        next_states_full = torch.cat(next_states, dim=1)
        actions_full = torch.cat(actions, dim=1)

        return states, actions, rewards, next_states, dones, states_full, next_states_full, actions_full


def train(args):
    agents, num_agents, action_sizes, action_low, action_high, state_sizes = get_env_info(
        env_name=args.env_name, max_steps=args.max_steps
    )
    env = create_single_env(args.env_name, max_steps=args.max_steps)

    agent_groups = [0] * num_agents
    if "tag" in args.env_name:
        agent_groups[num_agents - 1] = 1
    print(f"Agent Groups: {agent_groups}")

    maddpg = MADDPG(
        state_sizes=state_sizes,
        action_sizes=action_sizes,
        agent_groups=agent_groups,
        use_dag_attention=args.use_dag,
        hidden_sizes=(256, 256),
        actor_lr=args.lr_actor,
        critic_lr=args.lr_critic,
        gamma=args.gamma,
        action_low=action_low,
        action_high=action_high
    )

    buffer = ReplayBuffer(100000, num_agents, state_sizes, action_sizes)
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_str = "DAG" if args.use_dag else "MLP"
    #run_dir = os.path.join(args.save_dir, args.env_name, f"{mode_str}_{timestamp}")
    # 1. 先确定算法名字
    algo_name = "DAG" if args.use_dag else "MLP"

    # 2. 确定文件夹名字
    if args.run_name:
        # 优先使用命令行指定的自定义名字
        sub_dir_name = args.run_name
    else:
        # 备用方案：算法名 + 时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sub_dir_name = f"{algo_name}_{timestamp}"

    # 3. 拼接并创建路径
    # 注意：这里最好用 args.save_dir (通常是 ./runs)，如果没有定义 save_dir，写死 "./runs" 也没问题
    run_dir = os.path.join("./runs", args.env_name, sub_dir_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"📁 Training data will be saved to: {run_dir}")

    scores_deque = deque(maxlen=100)
    start_time = time.time()

    print(f"Start Training: {args.env_name} | Mode: {mode_str} | Agents: {num_agents}")

    for i_episode in range(1, args.episodes + 1):
        obs, _ = env.reset()
        episode_rewards = np.zeros(num_agents)
        #noise = max(0.05, 1.0 - i_episode / 15000.0)
        #noise = 0.01
        noise = max(0.05, 1.0 - i_episode / 30000.0)

        for step in range(args.max_steps):
            states = [np.array(obs[ag], dtype=np.float32) for ag in agents]
            actions_list = maddpg.act(states, add_noise=True, noise_scale=noise)
            actions_dict = {ag: act for ag, act in zip(agents, actions_list)}

            next_obs, rewards, terms, truncs, _ = env.step(actions_dict)
            next_states = [np.array(next_obs[ag], dtype=np.float32) for ag in agents]
            dones_list = [terms[ag] or truncs[ag] for ag in agents]
            rewards_list = [rewards[ag] for ag in agents]

            buffer.add(states, actions_list, rewards_list, next_states, dones_list)

            if buffer.size > args.batch_size:
                for agent_idx in range(num_agents):
                    sample = buffer.sample(args.batch_size)
                    maddpg.learn(sample, agent_idx)
                maddpg.update_targets()

            obs = next_obs
            episode_rewards += rewards_list

            if any(dones_list):
                break

        scores_deque.append(np.sum(episode_rewards))

        if i_episode % args.print_rate == 0:
            avg_score = np.mean(scores_deque)
            print(
                f"Ep {i_episode} | Avg Score: {avg_score:.2f} | Noise: {noise:.2f} | Time: {int(time.time() - start_time)}s")

        if i_episode % args.save_rate == 0:
            save_path = os.path.join(run_dir, "model.pt")
            maddpg.save(save_path)

    maddpg.save(os.path.join(run_dir, "final_model.pt"))
    env.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)