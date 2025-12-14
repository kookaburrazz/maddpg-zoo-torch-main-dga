import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
# 修改引用：导入 DGACritic 和 MLPCritic
from .networks import Actor, DGACritic, MLPCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPGAgent:
    """DDPG Agent with Actor and DAG-Attention Support"""

    def __init__(self, state_size, action_size,
                 agent_index=None,
                 all_state_sizes=None,
                 all_action_sizes=None,
                 agent_groups=None,
                 use_dag_attention=False,
                 hidden_sizes=(64, 64),
                 actor_lr=1e-4, critic_lr=1e-3, tau=1e-3,
                 centralized=False, total_state_size=None, total_action_size=None,
                 action_low=None, action_high=None):

        self.state_size = state_size
        self.action_size = action_size
        self.tau = tau
        self.centralized = centralized
        self.agent_index = agent_index
        self.use_dag_attention = use_dag_attention

        # Action bounds
        self.action_low = -1.0 if action_low is None else action_low
        self.action_high = 1.0 if action_high is None else action_high
        self.action_range = self.action_high - self.action_low

        # 1. Actor (不变)
        self.actor = Actor(state_size, action_size, hidden_sizes,
                           action_low=self.action_low,
                           action_high=self.action_high).to(device)
        self.actor_target = Actor(state_size, action_size, hidden_sizes,
                                  action_low=self.action_low,
                                  action_high=self.action_high).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # 2. Critic (修正后的逻辑)
        if self.use_dag_attention:
            # --- 分支 A: 使用 DGACritic ---
            self.critic = DGACritic(
                state_sizes=all_state_sizes,
                action_sizes=all_action_sizes,
                agent_groups=agent_groups,
                hidden_sizes=hidden_sizes
            ).to(device)
            self.critic_target = DGACritic(
                state_sizes=all_state_sizes,
                action_sizes=all_action_sizes,
                agent_groups=agent_groups,
                hidden_sizes=hidden_sizes
            ).to(device)

        elif centralized:
            # --- 分支 B: 使用 MLPCritic (旧版) ---
            # 确保 networks.py 里有 MLPCritic 类
            self.critic = MLPCritic(total_state_size, total_action_size, hidden_sizes).to(device)
            self.critic_target = MLPCritic(total_state_size, total_action_size, hidden_sizes).to(device)
        else:
            # --- 分支 C: 单智能体 Critic ---
            # 也可以复用 MLPCritic，只是维度变小
            self.critic = MLPCritic(state_size, action_size, hidden_sizes).to(device)
            self.critic_target = MLPCritic(state_size, action_size, hidden_sizes).to(device)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.hard_update(self.critic_target, self.critic)
        self.hard_update(self.actor_target, self.actor)

    # ... (其余方法如 act, learn 等保持不变，因为我们已经统一了接口) ...
    # 比如 act, act_target, learn, hard_update, soft_update, save, load
    # ...

    # 只需要确保 act 方法存在：
    def act(self, state, add_noise=True, noise_scale=1.0):
        state = torch.from_numpy(state).float().to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()
        if add_noise:
            scaled_noise = np.random.normal(0, noise_scale * self.action_range, size=action.shape)
            action += scaled_noise
        return np.clip(action, self.action_low, self.action_high)

    def act_target(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float().to(device)
        return self.actor_target(state)

    def learn(self, experiences, gamma=0.99):
        # 这里的代码完全可以直接使用你之前写的
        # 只要 Critic 的 forward 接口兼容 (都有 current_agent_idx 参数)，就能跑通
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            # 无论哪种 Critic，都传入 index。MLPCritic 会自动忽略它。
            Q_targets_next = self.critic_target(next_states, next_actions, self.agent_index)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.critic(states, actions, self.agent_index)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

        # Actor Update
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred, self.agent_index).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optimizer.step()

        self.soft_update(self.critic_target, self.critic)
        self.soft_update(self.actor_target, self.actor)

        return critic_loss.item(), actor_loss.item()

    def hard_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)