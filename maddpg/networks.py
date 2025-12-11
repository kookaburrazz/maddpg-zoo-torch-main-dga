"""
Neural Network architectures for DDPG
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def init_weights(module, output_layer=None, init_w=3e-3):
    """Initialize network weights using standard PyTorch initialization
    
    Args:
        module: PyTorch module to initialize
        output_layer: The output layer that should use uniform initialization
        init_w: Weight initialization range for output layer
    """
    if isinstance(module, nn.Linear):
        if module == output_layer:  # Output layer
            # Use uniform initialization for the final layer
            nn.init.uniform_(module.weight, -init_w, init_w)
            nn.init.uniform_(module.bias, -init_w, init_w)
        else:  # Hidden layers
            # Use Kaiming initialization for ReLU layers
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(module.bias)

def _init_weights_approx(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1.0)  # Stable gain
        nn.init.constant_(module.bias, 0)
    if hasattr(module, 'fc3') and module is module.fc3:  # Final layer
        nn.init.uniform_(module.weight, -3e-3, 3e-3)
        nn.init.constant_(module.bias, 0)


class Actor(nn.Module):
    """Actor (Policy) Model"""
    
    def __init__(self, state_size, action_size, hidden_sizes=(64, 64), init_w=3e-3, 
                 action_low=-1.0, action_high=1.0):
        """
        Initialize parameters and build model.
        
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_sizes (tuple): Sizes of hidden layers
            init_w (float): Final layer weight initialization
            action_low (float or array): Lower bound of the action space (default: -1.0)
            action_high (float or array): Upper bound of the action space (default: 1.0)
        """
        super(Actor, self).__init__()
        
        self.action_low = action_low
        self.action_high = action_high
        self.scale = (action_high - action_low) / 2.0
        self.bias = (action_high + action_low) / 2.0
        
        # Build the network
        self.fc1 = nn.Linear(state_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], action_size)
        
        # Initialize weights
        self.apply(lambda m: init_weights(m, self.fc3, init_w))
        
    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Output is in range [-1, 1]
        
        # Scale from [-1, 1] to [action_low, action_high]
        return self._scale_action(x)
    
    def _scale_action(self, action):
        """Scale action from [-1, 1] to [action_low, action_high]"""
        return self.scale * action  + self.bias

class Critic(nn.Module):
    """Critic (Value) Model"""

    def __init__(self, state_sizes, action_sizes, hidden_sizes=(64, 64), init_w=3e-3, head_dim=128):
        """
        Initialize parameters and build model with DGA.

        Args:
            state_sizes (list): List of state dimensions for each agent.
            action_sizes (list): List of action dimensions for each agent.
            hidden_sizes (tuple): Sizes of subsequent hidden layers.
            head_dim (int): The output dimension of the DGA Encoder (replaces the first layer).
        """
        super(Critic, self).__init__()

        # 1. 替换原始的 fc1 层，使用 DGA 模块作为 Encoder
        self.dga_encoder = DGACriticEncoder(
            state_sizes=state_sizes,
            action_sizes=action_sizes,
            head_dim=hidden_sizes[0]  # DGA的输出维度设置为Critic后续的第一层维度
        )

        # 2. 后续网络层保持不变，但输入维度变为 head_dim (即 hidden_sizes[0])
        # 注意：这里我们使用 hidden_sizes[0] 作为 DGA 的 head_dim
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], 1)

        # 3. 初始化权重 (保持原有的初始化逻辑，但 DGA 模块应在自身 __init__ 中初始化)
        self.apply(lambda m: init_weights(m, self.fc3, init_w))

    def forward(self, state, action, current_agent_idx=0):
        """
        Build a critic (value) network that maps (global state, global action) pairs -> Q-values

        Args:
            state: states_full (Global state observation, concatenated)
            action: actions_full (Global action, concatenated)
            current_agent_idx: The index of the agent whose Q-value is being calculated
        """

        # 1. 调用 DGA 模块作为输入编码器
        # DGA 模块在内部进行数据的拆解、解耦 Attention 和门控
        x = self.dga_encoder(state, action, current_agent_idx)

        # 2. 后续网络层保持不变
        x = F.relu(self.fc2(x))
        x = F.relu(x)  # 确保 fc2 之后有激活函数 (如果 fc2 之后是 fc3，则 fc2 之后应有激活函数)

        # 根据您原始的代码结构，fc2 之后直接接 fc3
        x = F.relu(x)  # 保持原有的 ReLU 激活，使 Critic 具有非线性

        # 3. 输出 Q-value
        return self.fc3(x)  # Output is Q-value




# Problem: TanhTransform in TransformedDistribution computes log_prob by inverting 
# the transform: atanh((action - bias) / scale). For your setup (bias=0.5, scale=0.5), 
# actions of 0 or 1 map to atanh(-1) or atanh(1), which are undefined (infinite in 
# theory, NaN in practice due to numerical limits).
# 
# Solution: Clamp the action to [-0.999999, 0.999999] before inverting the transform.
# This avoids the undefined values and the NaN in practice (edge cases).
# 
# Other Solution is adjust the action space definition 
class SafeTanhTransform(torch.distributions.transforms.TanhTransform):
    """Safe Tanh Transform"""

    def _inverse(self, y):
        """Inverse of the TanhTransform"""
        # Clamp to avoid exact -1 or 1
        y = torch.clamp(y, -0.999999, 0.999999)
        return torch.atanh(y)

class ApproxActor(nn.Module):
    """Approximate Actor Network"""

    def __init__(self, state_size, action_size, hidden_sizes=(64, 64), init_w=3e-3, 
                 action_low=-1.0, action_high=1.0):
        super(ApproxActor, self).__init__()
        
        # Build the network
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
        """Get the distribution of the action"""
        base_distribution = torch.distributions.Normal(mu, torch.exp(log_std))
        # tanh_transform = torch.distributions.transforms.TanhTransform(cache_size=1)
        tanh_transform = SafeTanhTransform(cache_size=1)
        scale_transform = torch.distributions.transforms.AffineTransform(self.bias, self.scale)
        squashed_and_scaled_dist = torch.distributions.TransformedDistribution(base_distribution, [tanh_transform, scale_transform])
        return squashed_and_scaled_dist, base_distribution

    def sample(self, state, deterministic=False):
        """Sample an action from the actor network"""
        mu, log_std = self.forward(state)
    
        if deterministic:
            action = torch.tanh(mu) * self.scale + self.bias
            return action, None, None
        
        dist, base_dist = self._get_dist(mu, log_std)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = base_dist.entropy().sum(-1, keepdim=True) 

        return action, log_prob, entropy
    
    def evaluate_actions(self, state, action):
        """Evaluate the log probability of actions"""
        mu, log_std = self.forward(state)
        
        dist, base_dist = self._get_dist(mu, log_std)

        # Old way of clamping the action
        # action = torch.clamp(action, self.action_low + 1e-6, self.action_high - 1e-6)

        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = base_dist.entropy().sum(-1, keepdim=True) # 

        return log_prob, entropy


class DGACriticEncoder(nn.Module):
    def __init__(self, state_sizes, action_sizes, hidden_sizes=(64, 64), init_w=3e-3, head_dim=128):
        # ... (参数定义和初始化，如 super(DGACriticEncoder, self).__init__())

        # ... (num_agents 和 head_dim 定义)

        # 1. 计算局部特征维度 (D_feat)
        # 注意：使用第一个智能体的维度作为标准（假设同构环境）
        D_feat = state_sizes[0] + action_sizes[0]
        self.D_feat = D_feat

        # --- 2. Q 投影层 (Query) ---
        self.W_Q = nn.Linear(D_feat, head_dim)

        # --- 3. K/V 解耦投影层和 Gating 层 ---

        # 自身 (Self): K/V
        self.W_K_self = nn.Linear(D_feat, head_dim)
        self.W_V_self = nn.Linear(D_feat, head_dim)

        # 队友 (Team): K/V + Gating
        self.W_K_team = nn.Linear(D_feat, head_dim)
        self.W_V_team = nn.Linear(D_feat, head_dim)
        # Gating 机制：将 Attention 输出映射到一个 [0, 1] 的过滤系数
        self.gating_team = nn.Sequential(
            nn.Linear(head_dim, head_dim),
            nn.Sigmoid()
        )

        # 对手 (Opponent): K/V + Gating
        self.W_K_opp = nn.Linear(D_feat, head_dim)
        self.W_V_opp = nn.Linear(D_feat, head_dim)
        self.gating_opp = nn.Sequential(
            nn.Linear(head_dim, head_dim),
            nn.Sigmoid()
        )

        # 4. 融合层 (输入是 3 个 Attention 头的拼接: 3 * head_dim)
        self.output_linear = nn.Linear(head_dim * 3, head_dim)

        # 5. 权重初始化（可选：如果 DGA 类独立于 Critic，需要手动应用初始化）
        self.apply(lambda m: init_weights(m, self.output_linear))

    def forward(self, states_full, actions_full, current_agent_idx=0):
        B = states_full.shape[0]  # Batch Size
        device = states_full.device

        # A. 数据拆解 (与骨架代码一致)
        states_list = torch.split(states_full, self.state_sizes, dim=1)
        actions_list = torch.split(actions_full, self.action_sizes, dim=1)
        local_features = [torch.cat([states_list[i], actions_list[i]], dim=1) for i in range(self.num_agents)]

        B = states_full.shape[0]
        device = states_full.device
        all_indices = torch.arange(self.num_agents, device=device)

        # 1. Query 投影 (Q)
        Q_i = self.W_Q(X[current_agent_idx])  # [B, head_dim]

        # --- 2. 角色拆分和特征提取 ---

        # 提取所有非当前 Agent 的特征
        other_indices = all_indices[all_indices != current_agent_idx]
        X_other = X[other_indices].transpose(0, 1)  # X_other: [B, N-1, D_feat]

        # 假设：您需要在外部传入或根据 current_agent_idx 计算出队友和对手的 Mask
        # 由于我们无法直接访问环境配置，这里使用一个通用的索引来模拟拆分：
        # 假设 Agent 0 是 Predator，Agent 1, 2, 3 是 Hiders。
        # 如果 current_agent_idx 是 1，那么队友是 2, 3，对手是 0。

        # 关键的拆分步骤：
        team_features = X_other[:, self.teammate_indices, :]  # [B, N_team, D_feat]
        opp_features = X_other[:, self.opponent_indices, :]  # [B, N_opp, D_feat]

        # --- C.1 自身 Attention (Self-Att) ---
        # ... (代码保持不变，att_self_out 已计算) ...

        # --- C.2 协作 Attention (Team-Att) ---

        if team_features.size(1) > 0:
            K_team = self.W_K_team(team_features)  # [B, N_team, head_dim]
            V_team = self.W_V_team(team_features)  # [B, N_team, head_dim]

            # score_team: [B, N_team]
            score_team = torch.bmm(Q_i.unsqueeze(1), K_team.transpose(1, 2)).squeeze(1) / (self.head_dim ** 0.5)
            alpha_team = F.softmax(score_team, dim=1).unsqueeze(2)  # [B, N_team, 1]

            team_att_output = torch.sum(alpha_team * V_team, dim=1)  # [B, head_dim]

            # 应用门控
            g_team = self.gating_team(team_att_output)
            att_team_out = team_att_output * g_team
        else:
            att_team_out = torch.zeros_like(Q_i)

        # --- C.3 竞争 Attention (Opponent-Att) ---

        if opp_features.size(1) > 0:
            # 逻辑与 C.2 相同，但使用 Opponent 的权重和门控
            K_opp = self.W_K_opp(opp_features)  # [B, N_opp, head_dim]
            V_opp = self.W_V_opp(opp_features)  # [B, N_opp, head_dim]

            # 评分: [B, 1, head_dim] * [B, head_dim, N_opp] -> [B, N_opp]
            score_opp = torch.bmm(Q_i.unsqueeze(1), K_opp.transpose(1, 2)).squeeze(1) / (self.head_dim ** 0.5)
            alpha_opp = F.softmax(score_opp, dim=1).unsqueeze(2)  # [B, N_opp, 1]

            opp_att_output = torch.sum(alpha_opp * V_opp, dim=1)  # [B, head_dim]

            # 应用门控
            g_opp = self.gating_opp(opp_att_output)
            att_opp_out = opp_att_output * g_opp  # Gated 结果
        else:
            att_opp_out = torch.zeros_like(Q_i)

        # ---------------------- D. 最终融合 (Fusion) ----------------------

        fused_output = torch.cat([att_self_out, att_team_out, att_opp_out], dim=-1)  # [B, 3 * head_dim]
        final_feature = F.relu(self.output_linear(fused_output))  # [B, head_dim]

        return final_feature