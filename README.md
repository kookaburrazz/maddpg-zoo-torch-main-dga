\# DAG-Attention MADDPG: Structured Multi-Agent Reinforcement Learning



This repository contains the PyTorch implementation of \*\*DAG-Attention MADDPG\*\*, a novel multi-agent reinforcement learning algorithm that improves credit assignment and coordination by introducing a Directed Acyclic Graph (DAG) attention mechanism into the Critic network.



The algorithm is evaluated on the \[PettingZoo MPE](https://pettingzoo.farama.org/environments/mpe/) environments, demonstrating significant improvements over the standard MADDPG baseline, particularly in tasks requiring complex coordination like `simple\_tag` (Predator-Prey) and `simple\_spread`.



\## Key Features



\* \*\*Novel Architecture\*\*: Incorporates DAG-based attention to model explicit dependencies between agents.

\* \*\*Solved "Lazy Agent" Problem\*\*: Effectively addresses the credit assignment problem in shared-reward environments.

\* \*\*Comprehensive Benchmarks\*\*: Ready-to-run scripts for `simple\_spread`, `simple\_tag`, and `simple\_adversary`.

\* \*\*Visualization\*\*: Built-in tools to generate GIF replays for qualitative analysis.



\## Installation



1\. Clone the repository:

&nbsp;   

&nbsp;   git clone \[https://github.com/YourUsername/DAG-Attention-MADDPG.git](https://github.com/YourUsername/DAG-Attention-MADDPG.git)

&nbsp;   cd DAG-Attention-MADDPG



2\. Install dependencies:

&nbsp;   

&nbsp;   pip install torch numpy pettingzoo imageio



\## Usage



We provide scripts for both \*\*Training\*\* (`train.py`) and \*\*Evaluation/Visualization\*\* (`run.py`).



\### 1. Train the Model



You can compare the \*\*Baseline (MLP)\*\* and \*\*Ours (DAG)\*\* using the following commands.



\#### Experiment A: Predator-Prey (`simple\_tag`)

\*The most visually distinct environment. DAG agents learn to surround the prey, while Baseline agents often simply chase.\*



&nbsp;   # Train Baseline (Standard MADDPG)

&nbsp;   python train.py --env-name simple\_tag\_v3 --run-name baseline\_tag\_60k --max-steps 60 --episodes 60000 --batch-size 1024 --lr-actor 0.005 --lr-critic 0.005



&nbsp;   # Train Ours (DAG-Attention)

&nbsp;   python train.py --env-name simple\_tag\_v3 --use-dag --run-name dag\_tag\_60k --max-steps 60 --episodes 60000 --batch-size 1024 --lr-actor 0.005 --lr-critic 0.005



\#### Experiment B: Cooperative Navigation (`simple\_spread`)

\*DAG agents demonstrate automatic role assignment without collision.\*



&nbsp;   # Train Baseline

&nbsp;   python train.py --env-name simple\_spread\_v3 --run-name baseline\_spread\_40k --max-steps 60 --episodes 40000 --batch-size 1024



&nbsp;   # Train Ours

&nbsp;   python train.py --env-name simple\_spread\_v3 --use-dag --run-name dag\_spread\_40k --max-steps 60 --episodes 40000 --batch-size 1024



\### 2. Evaluate \& Visualize (Generate GIFs)



After training, visualize the results to see the behavior difference.



&nbsp;   # Visualize Baseline

&nbsp;   python run.py --env-name simple\_tag\_v3 --model-path "./runs/simple\_tag\_v3/baseline\_tag\_60k/model.pt" --max-steps 60



&nbsp;   # Visualize DAG (Add --use-dag)

&nbsp;   python run.py --env-name simple\_tag\_v3 --use-dag --model-path "./runs/simple\_tag\_v3/dag\_tag\_60k/model.pt" --max-steps 60



\## Results



| Environment | Baseline Score | DAG Score (Ours) | Improvement |

| :--- | :--- | :--- | :--- |

| \*\*Simple Spread\*\* | ~ -180 | \*\*~ -105\*\* | \*\*+40%\*\* |

| \*\*Simple Tag\*\* | Low Coordination | \*\*Effective Encirclement\*\* | \*\*Significant\*\* |



\## File Structure



\* `train.py`: Main training loop.

\* `run.py`: Evaluation and GIF generation script.

\* `MADDPG.py`: Implementation of the MADDPG agent and the DAG-Attention Critic.

\* `utils/`: Helper functions for environment setup and buffer management.



\## Citation



If you find this code useful, please consider citing:



\[Your Name or Paper Title Here]

