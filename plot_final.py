import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Root folder containing your experiments
ROOT_DIR = "./runs_round2"
OUTPUT_DIR = "./plots"


def parse_log_file(filepath):
    """Parses a text log file for 'Ep X | Avg Score: Y'."""
    steps = []
    scores = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Regex to match: "Ep 50 | Avg Score: -123.45"
                match = re.search(r"Ep (\d+) \| Avg Score: ([\-\d\.]+)", line)
                if match:
                    steps.append(int(match.group(1)))
                    scores.append(float(match.group(2)))
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return steps, scores


def smooth(scalars, weight=0.9):
    """Smoothing function for cleaner plots."""
    if not scalars: return []
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def scan_and_plot():
    if not os.path.exists(ROOT_DIR):
        print(f"❌ Error: Folder '{ROOT_DIR}' not found.")
        return

    # Find all environments (subfolders in runs_round2)
    envs = [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]

    if not envs:
        print("❌ No environment folders found inside runs_round2.")
        return

    for env_name in envs:
        print(f"\n📊 Processing Environment: {env_name}...")
        env_path = os.path.join(ROOT_DIR, env_name)

        plt.figure(figsize=(10, 6))

        # Find all runs (sub-subfolders)
        runs = [d for d in os.listdir(env_path) if os.path.isdir(os.path.join(env_path, d))]

        has_data = False
        for run_name in runs:
            run_path = os.path.join(env_path, run_name)

            # Find the log file inside this run folder
            # We look for ANY .log or .txt file, or just parse the largest file
            files = [f for f in os.listdir(run_path) if f.endswith('.log') or f.endswith('.txt')]

            if not files:
                print(f"  ⚠️  No log file found in {run_name}")
                continue

            # Pick the largest log file (most likely to contain the full training)
            log_file = max([os.path.join(run_path, f) for f in files], key=os.path.getsize)
            print(f"  📖 Parsing {run_name} -> {os.path.basename(log_file)}")

            steps, scores = parse_log_file(log_file)

            if not steps:
                print(f"  ❌ No data found in file.")
                continue

            has_data = True

            # Determine Color and Label
            if "dag" in run_name.lower():
                color = 'red'
                label = "Ours (DAG-Attention)"
                zorder = 5
            else:
                color = 'blue'
                label = "Baseline (MADDPG)"
                zorder = 3

            # Plot
            plt.plot(steps, scores, color=color, alpha=0.15)  # Faint raw line
            plt.plot(steps, smooth(scores, 0.95), color=color, linewidth=2, label=label, zorder=zorder)  # Smoothed line

        if has_data:
            plt.title(f"Learning Curve: {env_name}", fontsize=14)
            plt.xlabel("Episodes")
            plt.ylabel("Average Reward")
            plt.grid(True, alpha=0.3)
            plt.legend()

            if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
            save_loc = os.path.join(OUTPUT_DIR, f"{env_name}_result.png")
            plt.savefig(save_loc, dpi=300)
            print(f"  ✅ Saved plot to {save_loc}")
        else:
            print("  ❌ No valid data found to plot for this environment.")

        plt.close()


if __name__ == "__main__":
    scan_and_plot()