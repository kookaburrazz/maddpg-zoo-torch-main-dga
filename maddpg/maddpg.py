import os
import torch
import torch.nn.functional as F

from .ddpg import DDPGAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MADDPG:
    """
    Multi-Agent DDPG controller (MADDPG).
    Coordinates multiple DDPG agents and centralized critic training.
    """

    def __init__(
        self,
        state_sizes,
        action_sizes,
        agent_groups,
        use_dag_attention=False,
        dag_no_gate=False,
        dag_no_decouple=False,
        hidden_sizes=(64, 64),
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=1e-3,
        action_low=-1.0,
        action_high=1.0,
    ):
        self.num_agents = len(state_sizes)
        self.state_sizes = state_sizes
        self.action_sizes = action_sizes
        self.agent_groups = agent_groups

        self.gamma = gamma
        self.tau = tau
        self.action_low = action_low
        self.action_high = action_high

        self.use_dag_attention = use_dag_attention
        self.dag_no_gate = dag_no_gate
        self.dag_no_decouple = dag_no_decouple

        # for standard centralized MLP critic
        self.total_state_size = sum(state_sizes)
        self.total_action_size = sum(action_sizes)

        self.agents = []
        for i in range(self.num_agents):
            agent = DDPGAgent(
                # basic
                state_size=state_sizes[i],
                action_size=action_sizes[i],
                hidden_sizes=hidden_sizes,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                tau=tau,
                centralized=True,

                # DAG params
                agent_index=i,
                all_state_sizes=state_sizes,
                all_action_sizes=action_sizes,
                agent_groups=agent_groups,
                use_dag_attention=use_dag_attention,
                dag_no_gate=dag_no_gate,
                dag_no_decouple=dag_no_decouple,

                # standard centralized dims
                total_state_size=self.total_state_size,
                total_action_size=self.total_action_size,
                action_low=action_low,
                action_high=action_high,
            )
            self.agents.append(agent)

    def act(self, states, add_noise=True, noise_scale=0.0):
        """
        states: list of numpy arrays, each [batch, state_dim] (vec env) or [state_dim] (single)
        returns: list of numpy arrays, each [batch, action_dim] or [action_dim]
        """
        return [agent.act(s, add_noise=add_noise, noise_scale=noise_scale)
                for agent, s in zip(self.agents, states)]

    def act_target(self, states):
        """
        states: list of torch tensors [batch, state_dim] or numpy arrays
        returns: list of torch tensors [batch, action_dim]
        """
        return [agent.act_target(s) for agent, s in zip(self.agents, states)]

    def learn(self, experiences, agent_idx):
        """
        experiences from ReplayBuffer.sample():
          states: list[Tensor] each [B, state_dim_i]
          actions: list[Tensor] each [B, action_dim_i]
          rewards: list[Tensor] each [B, 1] or [B]
          next_states: list[Tensor]
          dones: list[Tensor]
          states_full: Tensor [B, sum(state_dims)]
          next_states_full: Tensor [B, sum(state_dims)]
          actions_full: Tensor [B, sum(action_dims)]
        """
        states, actions, rewards, next_states, dones, states_full, next_states_full, actions_full = experiences

        current_agent = self.agents[agent_idx]
        r_i = rewards[agent_idx]
        d_i = dones[agent_idx]

        # ---------------- Critic update ----------------
        with torch.no_grad():
            next_actions_list = self.act_target(next_states)
            next_actions_full = torch.cat(next_actions_list, dim=1)

            if self.use_dag_attention:
                q_next = current_agent.critic_target(next_states_full, next_actions_full, agent_idx)
            else:
                q_next = current_agent.critic_target(next_states_full, next_actions_full)

            # make sure shapes broadcast
            if r_i.dim() == 1:
                r_i = r_i.unsqueeze(1)
            if d_i.dim() == 1:
                d_i = d_i.unsqueeze(1)

            q_target = r_i + self.gamma * q_next * (1 - d_i)

        if self.use_dag_attention:
            q_expected = current_agent.critic(states_full, actions_full, agent_idx)
        else:
            q_expected = current_agent.critic(states_full, actions_full)

        critic_loss = F.mse_loss(q_expected, q_target)

        current_agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(current_agent.critic.parameters(), 1.0)
        current_agent.critic_optimizer.step()

        # ---------------- Actor update ----------------
        actions_pred = []
        for j, agent in enumerate(self.agents):
            if j == agent_idx:
                actions_pred.append(current_agent.actor(states[j]))
            else:
                actions_pred.append(actions[j].detach())
        actions_full_pred = torch.cat(actions_pred, dim=1)

        if self.use_dag_attention:
            actor_loss = -current_agent.critic(states_full, actions_full_pred, agent_idx).mean()
        else:
            actor_loss = -current_agent.critic(states_full, actions_full_pred).mean()

        current_agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(current_agent.actor.parameters(), 0.5)
        current_agent.actor_optimizer.step()

        return critic_loss.item(), actor_loss.item()

    def update_targets(self):
        for agent in self.agents:
            self.soft_update(agent.actor_target, agent.actor)
            self.soft_update(agent.critic_target, agent.critic)

    def soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        models = {}
        for i, agent in enumerate(self.agents):
            models[f"agent_{i}_actor"] = agent.actor.state_dict()
            models[f"agent_{i}_critic"] = agent.critic.state_dict()
        torch.save(models, path)
        print(f"Models saved to {path}")

    def load(self, path):
        if not os.path.exists(path):
            print(f"Warning: No model file found at {path}")
            return

        models = torch.load(path, map_location=device, weights_only=False)
        for i, agent in enumerate(self.agents):
            ak = f"agent_{i}_actor"
            ck = f"agent_{i}_critic"
            if ak in models:
                agent.actor.load_state_dict(models[ak])
                agent.actor_target.load_state_dict(models[ak])
            if ck in models:
                agent.critic.load_state_dict(models[ck])
                agent.critic_target.load_state_dict(models[ck])
        print(f"All models loaded from {path}")
