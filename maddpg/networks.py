import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mlp(in_dim, hidden_sizes, out_dim, act=nn.ReLU, out_act=None):
    layers = []
    prev = in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), act()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    if out_act is not None:
        layers.append(out_act())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(256, 256), action_low=-1.0, action_high=1.0):
        super().__init__()
        self.net = _mlp(state_dim, hidden_sizes, action_dim, act=nn.ReLU, out_act=None)
        self.action_low = action_low
        self.action_high = action_high

    def forward(self, s):
        a = torch.tanh(self.net(s))  # [-1, 1]
        # scale to [low, high] if needed
        if self.action_low == -1.0 and self.action_high == 1.0:
            return a
        return (a + 1.0) * 0.5 * (self.action_high - self.action_low) + self.action_low


class MLPCritic(nn.Module):
    """
    Standard centralized critic: Q(s_all, a_all)
    NOTE: accepts an optional agent_idx and ignores it to be API-compatible.
    """

    def __init__(self, total_state_dim, total_action_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.q = _mlp(total_state_dim + total_action_dim, hidden_sizes, 1, act=nn.ReLU)

    def forward(self, states_full, actions_full, agent_idx=None):
        x = torch.cat([states_full, actions_full], dim=1)
        return self.q(x)


class _DotAttention(nn.Module):
    """
    Single-query dot-product attention over a set of tokens.
    query: [B, d]
    keys/values: [B, M, d]
    returns: summary [B, d]
    """

    def __init__(self, d_model):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, query, tokens):
        # tokens: [B, M, d]
        if tokens is None or tokens.size(1) == 0:
            return torch.zeros_like(query)

        q = self.q_proj(query).unsqueeze(1)      # [B, 1, d]
        k = self.k_proj(tokens)                  # [B, M, d]
        v = self.v_proj(tokens)                  # [B, M, d]

        attn_logits = (q * k).sum(dim=-1) * self.scale  # [B, M]
        w = torch.softmax(attn_logits, dim=-1).unsqueeze(-1)  # [B, M, 1]
        return (w * v).sum(dim=1)  # [B, d]


class DGACritic(nn.Module):
    """
    Decoupled Gated Attention critic:
      - builds per-agent token embeddings from (s_i, a_i)
      - decouples ally/adv sets based on agent_groups
      - gated fusion (or fixed 0.5 if dag_no_gate)
    """

    def __init__(
        self,
        state_sizes,
        action_sizes,
        agent_groups,
        hidden_sizes=(256, 256),
        d_model=128,
        dag_no_gate=False,
        dag_no_decouple=False,
    ):
        super().__init__()
        self.state_sizes = list(state_sizes)
        self.action_sizes = list(action_sizes)
        self.agent_groups = [list(g) for g in agent_groups]
        self.num_agents = len(self.state_sizes)

        self.d_model = d_model
        self.dag_no_gate = dag_no_gate
        self.dag_no_decouple = dag_no_decouple

        # slices for splitting full vectors
        self.state_splits = self.state_sizes
        self.action_splits = self.action_sizes

        # token embed per agent from concat(s_i, a_i)
        self.token_proj = nn.ModuleList([
            nn.Linear(self.state_sizes[i] + self.action_sizes[i], d_model)
            for i in range(self.num_agents)
        ])

        # attention modules (ally/adv)
        self.attn_ally = _DotAttention(d_model)
        self.attn_adv = _DotAttention(d_model)

        # gate
        self.gate = nn.Linear(2 * d_model, d_model)

        # output Q network
        # Input: [s_i, a_i, E_i] where E_i is fused context
        self.q_mlp = _mlp(self.state_sizes[0] + self.action_sizes[0] + d_model, hidden_sizes, 1, act=nn.ReLU)

        # NOTE: state/action dims can differ by agent; simplest is build per-agent q heads.
        # To keep it robust, create per-agent q heads.
        self.q_heads = nn.ModuleList([
            _mlp(self.state_sizes[i] + self.action_sizes[i] + d_model, hidden_sizes, 1, act=nn.ReLU)
            for i in range(self.num_agents)
        ])

    def _agent_group_id(self, agent_idx: int) -> int:
        for gid, g in enumerate(self.agent_groups):
            if agent_idx in g:
                return gid
        # fallback: everyone in one group
        return 0

    def _split_full(self, states_full, actions_full):
        # returns lists of tensors per agent
        s_list = list(torch.split(states_full, self.state_splits, dim=1))
        a_list = list(torch.split(actions_full, self.action_splits, dim=1))
        return s_list, a_list

    def forward(self, states_full, actions_full, current_agent_idx: int):
        """
        states_full: [B, sum(state_dims)]
        actions_full: [B, sum(action_dims)]
        current_agent_idx: int
        """
        s_list, a_list = self._split_full(states_full, actions_full)

        # build token embeddings for all agents
        tokens = []
        for i in range(self.num_agents):
            tok_in = torch.cat([s_list[i], a_list[i]], dim=1)  # [B, s_i+a_i]
            tok = torch.relu(self.token_proj[i](tok_in))       # [B, d]
            tokens.append(tok)

        i = current_agent_idx
        query = tokens[i]  # [B, d]

        # build sets
        others = [j for j in range(self.num_agents) if j != i]
        if self.dag_no_decouple:
            # one shared stream over all others
            all_tokens = torch.stack([tokens[j] for j in others], dim=1) if len(others) > 0 else None
            h_all = self.attn_ally(query, all_tokens)  # reuse module
            E = h_all
        else:
            gid = self._agent_group_id(i)
            ally = [j for j in others if self._agent_group_id(j) == gid]
            adv = [j for j in others if self._agent_group_id(j) != gid]

            ally_tokens = torch.stack([tokens[j] for j in ally], dim=1) if len(ally) > 0 else torch.zeros(
                (query.size(0), 0, self.d_model), device=query.device
            )
            adv_tokens = torch.stack([tokens[j] for j in adv], dim=1) if len(adv) > 0 else torch.zeros(
                (query.size(0), 0, self.d_model), device=query.device
            )

            h_ally = self.attn_ally(query, ally_tokens)
            h_adv = self.attn_adv(query, adv_tokens)

            if self.dag_no_gate:
                E = 0.5 * h_ally + 0.5 * h_adv
            else:
                z = torch.sigmoid(self.gate(torch.cat([h_ally, h_adv], dim=1)))  # [B, d]
                E = z * h_ally + (1.0 - z) * h_adv

        # per-agent head: Q_i(s_i, a_i, E)
        sa_i = torch.cat([s_list[i], a_list[i], E], dim=1)
        q = self.q_heads[i](sa_i)
        return q
