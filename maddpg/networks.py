import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def init_weights(module, output_layer=None, init_w=3e-3):
    """Standard weight initialization"""
    if isinstance(module, nn.Linear):
        if module == output_layer:
            nn.init.uniform_(module.weight, -init_w, init_w)
            nn.init.uniform_(module.bias, -init_w, init_w)
        else:
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(module.bias)


def _init_weights_approx(module):
    """Initialization for Approx Actor"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
        nn.init.constant_(module.bias, 0)
    if hasattr(module, 'fc3') and module is module.fc3:
        nn.init.uniform_(module.weight, -3e-3, 3e-3)
        nn.init.constant_(module.bias, 0)


# --- 基础 Actor ---
class Actor(nn.Module):
    """Standard Actor (Policy) Model"""

    def __init__(self, state_size, action_size, hidden_sizes=(64, 64), init_w=3e-3,
                 action_low=-1.0, action_high=1.0):
        super(Actor, self).__init__()
        self.action_low = action_low
        self.action_high = action_high
        self.scale = (action_high - action_low) / 2.0
        self.bias = (action_high + action_low) / 2.0

        self.fc1 = nn.Linear(state_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], action_size)

        self.apply(lambda m: init_weights(m, self.fc3, init_w))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return self.scale * x + self.bias


# --- Approx Actor (恢复这个类以修复 import 错误) ---
class SafeTanhTransform(torch.distributions.transforms.TanhTransform):
    def _inverse(self, y):
        y = torch.clamp(y, -0.999999, 0.999999)
        return torch.atanh(y)


class ApproxActor(nn.Module):
    """Approximate Actor Network"""

    def __init__(self, state_size, action_size, hidden_sizes=(64, 64), init_w=3e-3,
                 action_low=-1.0, action_high=1.0):
        super(ApproxActor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], action_size * 2)

        self.action_low = action_low
        self.action_high = action_high
        self.scale = (action_high - action_low) / 2.0
        self.bias = (action_high + action_low) / 2.0
        self.apply(_init_weights_approx)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu, log_std = self.fc3(x).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mu, log_std

    def _get_dist(self, mu, log_std):
        base_distribution = torch.distributions.Normal(mu, torch.exp(log_std))
        tanh_transform = SafeTanhTransform(cache_size=1)
        scale_transform = torch.distributions.transforms.AffineTransform(self.bias, self.scale)
        return torch.distributions.TransformedDistribution(base_distribution,
                                                           [tanh_transform, scale_transform]), base_distribution

    def sample(self, state, deterministic=False):
        mu, log_std = self.forward(state)
        if deterministic:
            return torch.tanh(mu) * self.scale + self.bias, None, None
        dist, base_dist = self._get_dist(mu, log_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = base_dist.entropy().sum(-1, keepdim=True)
        return action, log_prob, entropy


# --- 核心组件：DGA Encoder (完全修复版) ---
class DGACriticEncoder(nn.Module):
    def __init__(self, state_sizes, action_sizes, agent_groups, head_dim=128):
        super(DGACriticEncoder, self).__init__()
        self.state_sizes = state_sizes
        self.action_sizes = action_sizes
        self.num_agents = len(state_sizes)
        self.agent_groups = agent_groups
        self.head_dim = head_dim

        # Assume uniform feature size for simplicity (sum of state+action dims)
        # 如果你的 Agent state 维度不同，这里可能需要 Padding 或 Projection
        self.D_feat = state_sizes[0] + action_sizes[0]

        self.W_Q = nn.Linear(self.D_feat, head_dim)
        self.W_Self = nn.Linear(self.D_feat, head_dim)

        self.W_K_team = nn.Linear(self.D_feat, head_dim)
        self.W_V_team = nn.Linear(self.D_feat, head_dim)
        self.gating_team = nn.Sequential(nn.Linear(head_dim, head_dim), nn.Sigmoid())

        self.W_K_opp = nn.Linear(self.D_feat, head_dim)
        self.W_V_opp = nn.Linear(self.D_feat, head_dim)
        self.gating_opp = nn.Sequential(nn.Linear(head_dim, head_dim), nn.Sigmoid())

        self.output_linear = nn.Linear(head_dim * 3, head_dim)

    def forward(self, states_full, actions_full, current_agent_idx=0):
        # 1. 拆分 Full State 和 Full Action
        # states_full shape: (batch, sum(state_sizes))
        states_list = torch.split(states_full, self.state_sizes, dim=1)
        actions_list = torch.split(actions_full, self.action_sizes, dim=1)

        # 2. 拼接每个 Agent 的特征 (State + Action)
        X_list = [torch.cat([s, a], dim=1) for s, a in zip(states_list, actions_list)]

        # 3. 找到当前 Agent 的特征
        curr_feat = X_list[current_agent_idx]

        # 4. 确定当前 Agent 的分组 ID
        curr_group_id = None
        for idx, g in enumerate(self.agent_groups):
            if current_agent_idx in g:
                curr_group_id = idx
                break

        if curr_group_id is None:
            raise ValueError(f"Agent {current_agent_idx} not found in any group!")

        # 5. 分离队友和敌人
        team_feats_list = []
        opp_feats_list = []
        for i in range(self.num_agents):
            if i == current_agent_idx: continue

            # 修复逻辑：判断 i 是否在当前组的名单里
            # self.agent_groups[curr_group_id] 是一个列表，例如 [0, 1, 2]
            if i in self.agent_groups[curr_group_id]:
                team_feats_list.append(X_list[i])
            else:
                opp_feats_list.append(X_list[i])

        # 6. Self Attention 编码
        feat_self = F.relu(self.W_Self(curr_feat))
        Q = self.W_Q(curr_feat).unsqueeze(1)  # (batch, 1, head_dim)

        # 7. Team Attention
        if team_feats_list:
            X_team = torch.stack(team_feats_list, dim=1)  # (batch, num_team, D_feat)
            K = self.W_K_team(X_team)
            V = self.W_V_team(X_team)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_weights = F.softmax(scores, dim=-1)
            feat_team = torch.matmul(attn_weights, V).squeeze(1)
            feat_team = feat_team * self.gating_team(feat_team)
        else:
            feat_team = torch.zeros_like(feat_self)

        # 8. Opponent Attention
        if opp_feats_list:
            X_opp = torch.stack(opp_feats_list, dim=1)
            K = self.W_K_opp(X_opp)
            V = self.W_V_opp(X_opp)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_weights = F.softmax(scores, dim=-1)
            feat_opp = torch.matmul(attn_weights, V).squeeze(1)
            feat_opp = feat_opp * self.gating_opp(feat_opp)
        else:
            feat_opp = torch.zeros_like(feat_self)

        # 9. 输出
        return self.output_linear(torch.cat([feat_self, feat_team, feat_opp], dim=1))


# --- Critic 类定义 ---
class DGACritic(nn.Module):
    """New DAG-Attention Critic"""

    def __init__(self, state_sizes, action_sizes, agent_groups, hidden_sizes=(64, 64),
                 init_w=3e-3, head_dim=128):
        super(DGACritic, self).__init__()
        self.dga_encoder = DGACriticEncoder(state_sizes, action_sizes, agent_groups, head_dim)
        self.fc2 = nn.Linear(head_dim, hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], 1)
        self.apply(lambda m: init_weights(m, self.fc3, init_w))

    def forward(self, state, action, current_agent_idx=0):
        # 这里的 state, action 其实是 full_state, full_action
        x = self.dga_encoder(state, action, current_agent_idx)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MLPCritic(nn.Module):
    """Standard MLP Critic"""

    def __init__(self, total_state_size, total_action_size, hidden_sizes=(64, 64), init_w=3e-3):
        super(MLPCritic, self).__init__()
        self.fc1 = nn.Linear(total_state_size + total_action_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], 1)
        self.apply(lambda m: init_weights(m, self.fc3, init_w))

    def forward(self, state, action, current_agent_idx=None):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# --- 别名定义 ---
Critic = MLPCritic