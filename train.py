import torch
import numpy as np
import argparse
import os
import time
from datetime import datetime
from collections import deque

from maddpg.maddpg import MADDPG
from utils.env import create_single_env, ENV_MAP, get_env_info

def parse_args():
    parser = argparse.ArgumentParser("MADDPG Training with DAG Attention")
    
    # Environment
    parser.add_argument("--env-name", type=str, default="simple_spread_v3", choices=list(ENV_MAP.keys()))
    parser.add_argument("--max-steps", type=int, default=25, help="Max steps per episode")
    
    # Training
    parser.add_argument("--episodes", type=int, default=10000, help="Total training episodes")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr-actor", type=float, default=1e-4)
    parser.add_argument("--lr-critic", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    
    # DAG Attention Specific
    parser.add_argument("--use-dag", action="store_true", default=False, 
                       help="Enable DAG-Attention Critic")
    
    # Saving & Logging
    parser.add_argument("--save-dir", type=str, default="./runs", help="Directory to save models")
    parser.add_argument("--save-rate", type=int, default=1000, help="Save model every N episodes")
    parser.add_argument("--print-rate", type=int, default=100, help="Print progress every N episodes")
    
    return parser.parse_args()

class ReplayBuffer:
    def __init__(self, capacity, num_agents, state_sizes, action_sizes):
        self.capacity = capacity
        self.num_agents = num_agents
        self.ptr = 0
        self.size = 0
        
        # Initialize buffers
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
        
        states = [torch.tensor(self.states[i][idx], dtype=torch.float32).cuda() for i in range(self.num_agents)]
        actions = [torch.tensor(self.actions[i][idx], dtype=torch.float32).cuda() for i in range(self.num_agents)]
        rewards = [torch.tensor(self.rewards[i][idx], dtype=torch.float32).cuda() for i in range(self.num_agents)]
        next_states = [torch.tensor(self.next_states[i][idx], dtype=torch.float32).cuda() for i in range(self.num_agents)]
        dones = [torch.tensor(self.dones[i][idx], dtype=torch.float32).cuda() for i in range(self.num_agents)]
        
        # Concat for centralized critic
        states_full = torch.cat(states, dim=1)
        next_states_full = torch.cat(next_states, dim=1)
        actions_full = torch.cat(actions, dim=1)
        
        return states, actions, rewards, next_states, dones, states_full, next_states_full, actions_full

def train(args):
    # 1. Setup Environment
    agents, num_agents, action_sizes, action_low, action_high, state_sizes = get_env_info(
        env_name=args.env_name, max_steps=args.max_steps
    )
    env = create_single_env(args.env_name, max_steps=args.max_steps)

    # 2. Define Groups (Simple Logic)
    # 修正逻辑：我们需要一个长度为 num_agents 的列表，每个位置代表该 agent 的组 ID
    # 例如 simple_spread (3 agents) -> [0, 0, 0] (大家都是组0，完全合作)
    # 例如 simple_tag (3 pred, 1 prey) -> [0, 0, 0, 1] (前三个是组0，最后一个是组1)

    agent_groups = [0] * num_agents  # 默认所有人都是组 0

    if "tag" in args.env_name:
        # 如果是捕食者游戏，最后一个智能体是猎物，设为组 1
        # 前面的保持为 0
        agent_groups[num_agents - 1] = 1

    print(f"Agent Groups: {agent_groups}")  # 打印出来确认一下，应该是 [0, 0, 0] 这种

    # 3. Initialize MADDPG
    maddpg = MADDPG(
        state_sizes=state_sizes,
        action_sizes=action_sizes,
        agent_groups=agent_groups,
        use_dag_attention=args.use_dag, # <--- 开关
        hidden_sizes=(64, 64),
        actor_lr=args.lr_actor,
        critic_lr=args.lr_critic,
        gamma=args.gamma
    )
    
    # 4. Setup Buffer & Logging
    buffer = ReplayBuffer(100000, num_agents, state_sizes, action_sizes)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_str = "DAG" if args.use_dag else "MLP"
    run_dir = os.path.join(args.save_dir, args.env_name, f"{mode_str}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    scores_deque = deque(maxlen=100)
    start_time = time.time()
    
    print(f"Start Training: {args.env_name} | Mode: {mode_str} | Agents: {num_agents}")

    # 5. Training Loop
    for i_episode in range(1, args.episodes + 1):
        obs, _ = env.reset()
        episode_rewards = np.zeros(num_agents)
        
        # Noise decay
        noise = max(0.05, 1.0 - i_episode / 2000.0) # Decay over 2000 episodes
        
        for step in range(args.max_steps):
            # 1. Action
            states = [np.array(obs[ag], dtype=np.float32) for ag in agents]
            actions_list = maddpg.act(states, add_noise=True, noise_scale=noise)
            actions_dict = {ag: act for ag, act in zip(agents, actions_list)}
            
            # 2. Step
            next_obs, rewards, terms, truncs, _ = env.step(actions_dict)
            next_states = [np.array(next_obs[ag], dtype=np.float32) for ag in agents]
            dones_list = [terms[ag] or truncs[ag] for ag in agents]
            rewards_list = [rewards[ag] for ag in agents]
            
            # 3. Store
            buffer.add(states, actions_list, rewards_list, next_states, dones_list)
            
            # 4. Update
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
        
        # Logging
        if i_episode % args.print_rate == 0:
            avg_score = np.mean(scores_deque)

if __name__ == "__main__":
    args = parse_args()
    train(args)