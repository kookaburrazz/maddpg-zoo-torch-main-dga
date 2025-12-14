import os

import torch
import torch.nn.functional as F

from .ddpg import DDPGAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MADDPG:
    """
    Multi-Agent Deep Deterministic Policy Gradient (MADDPG) implementation.
    Orchestrates multiple DDPG agents and handles the centralized training logic.
    """

    def __init__(self, state_sizes, action_sizes, agent_groups, use_dag_attention=False,
                 hidden_sizes=(64, 64), actor_lr=1e-4, critic_lr=1e-3,
                 gamma=0.99, tau=1e-3, action_low=-1.0, action_high=1.0):
        """
        Initialize the MADDPG controller.

        Args:
            state_sizes (list): List of state dimensions for each agent.
            action_sizes (list): List of action dimensions for each agent.
            agent_groups (list of lists): Grouping of agents (e.g., [[0,1,2], [3,4]]) for DAG attention.
            use_dag_attention (bool): Whether to use DAG-based Attention mechanism in Critic.
            hidden_sizes (tuple): Hidden layer sizes for networks.
            ...
        """
        self.num_agents = len(state_sizes)
        self.state_sizes = state_sizes
        self.action_sizes = action_sizes
        self.gamma = gamma
        self.tau = tau
        self.action_low = action_low
        self.action_high = action_high

        # Flag to switch between standard MADDPG and DAG-Attention MADDPG
        self.use_dag_attention = use_dag_attention

        # Calculate total state and action sizes (used for standard Critic)
        self.total_state_size = sum(state_sizes)
        self.total_action_size = sum(action_sizes)

        # Create agents
        self.agents = []
        for i in range(self.num_agents):
            agent = DDPGAgent(
                # --- Basic DDPG Params ---
                state_size=state_sizes[i],
                action_size=action_sizes[i],
                hidden_sizes=hidden_sizes,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                tau=tau,
                centralized=True,

                # --- DAG-Attention Specific Params ---
                agent_index=i,  # ID of this agent (0 to N-1)
                all_state_sizes=state_sizes,  # List of all state sizes
                all_action_sizes=action_sizes,  # List of all action sizes
                agent_groups=agent_groups,  # Friend/Foe grouping
                use_dag_attention=use_dag_attention,

                # --- Standard MADDPG Params ---
                total_state_size=self.total_state_size,
                total_action_size=self.total_action_size,
                action_low=action_low,
                action_high=action_high
            )
            self.agents.append(agent)

    def act(self, states, add_noise=True, noise_scale=0.0):
        """
        Get actions from all agents based on current policy (for interaction).

        Args:
            states (list): List of state arrays (one per agent).
            add_noise (bool): Whether to add noise for exploration.
            noise_scale (float): Scale factor for noise.
        """
        actions = [agent.act(state, add_noise, noise_scale)
                   for agent, state in zip(self.agents, states)]
        return actions

    def act_target(self, states):
        """
        Get actions from all agents based on TARGET policies (for training).

        Args:
            states (list of tensors): Batch of states for each agent.

        Returns:
            actions (list of tensors): List of action tensors for each agent.
        """
        actions = [agent.act_target(state) for agent, state in zip(self.agents, states)]
        return actions

    def learn(self, experiences, agent_idx):
        """
        Update policy and value parameters for a specific agent.

        Args:
            experiences (tuple): Tuple containing unpacked batch data.
            agent_idx (int): The index of the agent currently being updated.
        """
        # Unpack experiences
        # states, next_states: list of tensors [batch, dim]
        # states_full, next_states_full: tensor [batch, total_dim] (Only used for standard Critic)
        states, actions, rewards, next_states, dones, states_full, next_states_full, actions_full = experiences

        current_agent = self.agents[agent_idx]
        agent_rewards = rewards[agent_idx]
        agent_dones = dones[agent_idx]

        # ---------------------------- Update Centralized Critic ---------------------------- #
        with torch.no_grad():
            # Get next actions from target actors
            next_actions_list = self.act_target(next_states)
            next_actions_full = torch.cat(next_actions_list, dim=1)

            # Compute Q_targets_next
            if self.use_dag_attention:
                # DAG Critic needs to know WHICH agent is looking (to apply mask)
                # Pass agent_idx to the critic_target
                Q_targets_next = current_agent.critic_target(next_states_full, next_actions_full, agent_idx)
            else:
                # Standard Critic takes concatenated vectors
                Q_targets_next = current_agent.critic_target(next_states_full, next_actions_full)

            # Compute Q targets (Bellman equation)
            Q_targets = agent_rewards + (self.gamma * Q_targets_next * (1 - agent_dones))

        # Compute Q_expected (Current Q)
        if self.use_dag_attention:
            Q_expected = current_agent.critic(states_full, actions_full, agent_idx)
        else:
            Q_expected = current_agent.critic(states_full, actions_full)

        # Critic Loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        current_agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(current_agent.critic.parameters(), 1.0)
        current_agent.critic_optimizer.step()

        # ---------------------------- Update Actor ---------------------------- #

        # Calculate policy gradient
        # We need to reconstruct the action list, replacing the current agent's action
        # with the one predicted by its CURRENT actor network.
        actions_pred = []
        for i, agent in enumerate(self.agents):
            if i == agent_idx:
                # Use current actor for the agent being updated (tracking gradients)
                actions_pred.append(current_agent.actor(states[i]))
            else:
                # Use sampled actions for other agents (detached)
                actions_pred.append(actions[i].detach())

        actions_full_pred = torch.cat(actions_pred, dim=1)

        # Compute Actor Loss using the Critic
        if self.use_dag_attention:
            actor_loss = -current_agent.critic(states_full, actions_full_pred, agent_idx).mean()
        else:
            actor_loss = -current_agent.critic(states_full, actions_full_pred).mean()

        # Minimize the loss
        current_agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(current_agent.actor.parameters(), 0.5)
        current_agent.actor_optimizer.step()

        return critic_loss.item(), actor_loss.item()

    def update_targets(self):
        """
        Soft update target networks for all agents.
        This should be called after all agents have been updated.
        """
        for agent in self.agents:
            self.soft_update(agent.actor_target, agent.actor)
            self.soft_update(agent.critic_target, agent.critic)

    def soft_update(self, target, source):
        """
        Soft update model parameters.
        θ_target = τ*θ_source + (1 - τ)*θ_target
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path):
        """Save all agent models to a single file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        models_dict = {}
        for i, agent in enumerate(self.agents):
            models_dict[f'agent_{i}_actor'] = agent.actor.state_dict()
            models_dict[f'agent_{i}_critic'] = agent.critic.state_dict()
        torch.save(models_dict, path)
        print(f"Models saved to {path}")

    def load(self, path):
        """Load all agent models from a single file."""
        if not os.path.exists(path):
            print(f"Warning: No model file found at {path}")
            return

        models_dict = torch.load(path, map_location=device, weights_only=False)

        for i, agent in enumerate(self.agents):
            actor_key = f'agent_{i}_actor'
            critic_key = f'agent_{i}_critic'

            if actor_key in models_dict:
                agent.actor.load_state_dict(models_dict[actor_key])
                agent.actor_target.load_state_dict(models_dict[actor_key])

            if critic_key in models_dict:
                agent.critic.load_state_dict(models_dict[critic_key])
                agent.critic_target.load_state_dict(models_dict[critic_key])

        print(f"All models loaded from {path}")