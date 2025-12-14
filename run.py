"""
Script to run trained agents in various environments (Updated for DAG-Attention)
"""
import torch
import numpy as np
import argparse
import os
import imageio
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

# 确保这里的引用路径正确，根据你的实际文件结构调整
from maddpg.maddpg import MADDPG
# from maddpg import MADDPGApprox # 如果你还没实现这个类，先注释掉

from utils.env import create_single_env, ENV_MAP, get_env_info

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,
                       default="runs/simple_spread_v3/...", # 替换为你实际的模型路径
                       help="Path to the trained model file")
    parser.add_argument("--env-name", type=str, default="simple_spread_v3",
                       choices=list(ENV_MAP.keys()),
                       help="Name of the environment to use")
    parser.add_argument("--algo", type=str, default="MADDPG", choices=["MADDPG", "MADDPGApprox"],
                       help="Algorithm to use")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=25, help="Maximum steps per episode")
    parser.add_argument("--output-dir", type=str, default="./gifs", help="Directory to save outputs")
    parser.add_argument("--is-parallel", action="store_true", help="Parallel environment")
    parser.add_argument("--create-gif", action="store_true", default=True,
                       help="Create GIF of episodes")
    parser.add_argument("--episode-separator", type=float, default=1.0,
                       help="Duration in seconds for the black frame between episodes")

    # --- 新增参数 ---
    parser.add_argument("--use-dag", action="store_true", default=False,
                       help="Use DAG-Attention Critic (Must match trained model)")

    return parser.parse_args()

def add_text_to_frame(frame, text):
    """Add simple text to a frame using PIL."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("Arial", 16)
    except IOError:
        font = ImageFont.load_default()
    position = (10, 10)
    draw.text(position, text, font=font, fill=(0, 0, 0)) # 黑色文字
    return np.array(img)

def run(args):
    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.env_name}/{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Get environment information
    agents, num_agents, action_sizes, action_low, action_high, state_sizes = get_env_info(
        env_name=args.env_name,
        max_steps=args.max_steps,
        apply_padding=args.is_parallel
    )

    # --- 关键修复：定义 Agent Groups ---
    # 如果是 simple_spread (合作导航)，通常所有 Agent 是一组
    # 如果是 simple_tag (捕食者游戏)，通常 前N-1个是捕食者，最后1个是猎物
    # 这里为了跑通，假设如果是 spread 则全员一组，其他情况默认全员一组（你可以根据需要修改）
    if "spread" in args.env_name:
        agent_groups = [list(range(num_agents))] # e.g. [[0, 1, 2]]
    elif "tag" in args.env_name:
        # 简单假设：前 N-1 是组0，最后一个是组1
        adversaries = list(range(num_agents - 1))
        good_agents = [num_agents - 1]
        agent_groups = [adversaries, good_agents]
    else:
        # 默认分组
        agent_groups = [list(range(num_agents))]

    print(f"Agent Groups: {agent_groups}")

    # Create environment
    render_mode = "rgb_array" if args.create_gif else None
    env = create_single_env(
        env_name=args.env_name,
        max_steps=args.max_steps,
        render_mode=render_mode,
        apply_padding=args.is_parallel
    )

    # Create MADDPG agent
    if args.algo == "MADDPG":
        maddpg = MADDPG(
            state_sizes=state_sizes,
            action_sizes=action_sizes,
            agent_groups=agent_groups,      # <--- 必须传入
            use_dag_attention=args.use_dag, # <--- 必须传入
            hidden_sizes=(64, 64),
            action_low=action_low,
            action_high=action_high
        )
    elif args.algo == "MADDPGApprox":
        # 假设 Approximate 类也更新了接口
        # maddpg = MADDPGApprox(...)
        pass
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    # Load trained model
    if os.path.exists(args.model_path):
        try:
            maddpg.load(args.model_path)
            print(f"Loaded model from {args.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Tip: Ensure --use-dag flag matches how the model was trained.")
            return
    else:
        print(f"Warning: No model found at {args.model_path}, using random policies")

    # Track episode statistics
    all_episode_rewards = []
    all_frames = [] if args.create_gif else None

    # Run episodes
    for episode in range(1, args.episodes + 1):
        observations, _ = env.reset()
        episode_rewards = np.zeros(len(agents))
        done = False
        step = 0

        episode_frames = [] if args.create_gif else None

        while not done and step < args.max_steps:
            # Get states
            states = [np.array(observations[agent], dtype=np.float32) for agent in agents]

            # Get actions (Evaluation mode: add_noise=False)
            actions_list = maddpg.act(states, add_noise=False)

            # Convert to dict
            actions = {agent: action for agent, action in zip(agents, actions_list)}

            # Step
            next_observations, rewards, terminations, truncations, _ = env.step(actions)
            dones = [terminations[agent] or truncations[agent] for agent in agents]
            done = any(dones)

            observations = next_observations
            episode_rewards += np.array(list(rewards.values()))

            # Capture frame
            if args.create_gif:
                try:
                    frame = env.render()
                    text = f"Ep {episode} - Step {step}"
                    labeled_frame = add_text_to_frame(frame, text)
                    episode_frames.append(labeled_frame)
                    all_frames.append(labeled_frame)
                except Exception as e:
                    print(f"Render error: {e}")

            step += 1

        all_episode_rewards.append(episode_rewards)
        print(f"Episode {episode}, Total Reward: {np.sum(episode_rewards):.2f}")

        # Add separator frames
        if args.create_gif and episode < args.episodes and episode_frames:
            black_frame = np.zeros_like(episode_frames[0])
            for _ in range(int(args.episode_separator * 10)):
                all_frames.append(black_frame)

    # Save GIF
    if args.create_gif and all_frames:
        gif_path = os.path.join(output_dir, f"eval_{args.env_name}_{args.algo}.gif")
        imageio.mimsave(gif_path, all_frames, duration=0.1)
        print(f"Saved GIF to {gif_path}")

    env.close()

if __name__ == "__main__":
    args = parse_args()
    run(args)