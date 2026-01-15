import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from .networks import Actor, DGACritic, MLPCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPGAgent:
    """
    A single DDPG agent used by MADDPG controller.
    Supports:
      - Standard centralized MLP critic
      - DAG-Attention critic (DGACritic)
    """

    def __init__(
        self,
        state_size,
        action_size,
        agent_index=None,
        all_state_sizes=None,
        all_action_sizes=None,
        agent_groups=None,
        use_dag_attention=False,
        dag_no_gate=False,
        dag_no_decouple=False,
        hidden_sizes=(64, 64),
        actor_lr=1e-4,
        critic_lr=1e-3,
        tau=1e-3,
        centralized=False,
        total_state_size=None,
        total_action_size=None,
        action_low=None,
        action_high=None,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.agent_index = agent_index
        self.tau = tau
        self.centralized = centralized
        self.use_dag_attention = use_dag_attention

        # action bounds
        self.action_low = -1.0 if action_low is None else action_low
        self.action_high = 1.0 if action_high is None else action_high
        self.action_range = self.action_high - self.action_low

        # actor
        self.actor = Actor(state_size, action_size, hidden_sizes, self.action_low, self.action_high).to(device)
        self.actor_target = Actor(state_size, action_size, hidden_sizes, self.action_low, self.action_high).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # critic
        if self.use_dag_attention:
            assert all_state_sizes is not None and all_action_sizes is not None and agent_groups is not None, \
                "DGACritic requires all_state_sizes/all_action_sizes/agent_groups"
            self.critic = DGACritic(
                state_sizes=all_state_sizes,
                action_sizes=all_action_sizes,
                agent_groups=agent_groups,
                hidden_sizes=hidden_sizes,
                dag_no_gate=dag_no_gate,
                dag_no_decouple=dag_no_decouple,
            ).to(device)
            self.critic_target = DGACritic(
                state_sizes=all_state_sizes,
                action_sizes=all_action_sizes,
                agent_groups=agent_groups,
                hidden_sizes=hidden_sizes,
                dag_no_gate=dag_no_gate,
                dag_no_decouple=dag_no_decouple,
            ).to(device)
        else:
            if centralized:
                assert total_state_size is not None and total_action_size is not None
                self.critic = MLPCritic(total_state_size, total_action_size, hidden_sizes).to(device)
                self.critic_target = MLPCritic(total_state_size, total_action_size, hidden_sizes).to(device)
            else:
                self.critic = MLPCritic(state_size, action_size, hidden_sizes).to(device)
                self.critic_target = MLPCritic(state_size, action_size, hidden_sizes).to(device)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic_target, self.critic)

    # -------- act helpers (supports batched numpy) --------
    def act(self, state, add_noise=True, noise_scale=0.0):
        """
        state: np.ndarray [B, state_dim] OR [state_dim]
        returns: np.ndarray [B, action_dim] OR [action_dim]
        """
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)

        state_t = torch.from_numpy(state).float().to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()
        self.actor.train()

        if add_noise and noise_scale > 0.0:
            noise = np.random.normal(0.0, noise_scale * self.action_range, size=action.shape).astype(np.float32)
            action = action + noise

        return np.clip(action, self.action_low, self.action_high)

    def act_target(self, state):
        """
        state: torch.Tensor [B, state_dim] OR np.ndarray
        returns: torch.Tensor [B, action_dim]
        """
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(np.array(state, dtype=np.float32))
        state = state.float().to(device)
        return self.actor_target(state)

    # Optional single-agent learn (not used by your MADDPG controller, but kept safe)
    def learn(self, experiences, gamma=0.99):
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_next = self.critic_target(next_states, next_actions, self.agent_index)
            q_target = rewards + gamma * q_next * (1 - dones)

        q_expected = self.critic(states, actions, self.agent_index)
        critic_loss = F.mse_loss(q_expected, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred, self.agent_index).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        self._soft_update(self.critic_target, self.critic)
        self._soft_update(self.actor_target, self.actor)

        return critic_loss.item(), actor_loss.item()

    def _hard_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(sp.data)

    def _soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * sp.data + (1.0 - self.tau) * tp.data)
