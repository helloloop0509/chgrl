from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    lr_policy: float = 3e-4
    lr_lambda: float = 1e-2
    target_kl: float = 0.02
    max_grad_norm: float = 0.5
    ppo_epochs: int = 10
    minibatch_size: int = 256


class SimpleGraphEncoder(nn.Module):
    """
    Minimal graph encoder without external dependencies.
    graph_state:
      - x:        [N, node_feat_dim]
      - adj:      [N, N] binary/weighted adjacency
      - graph_id: [N] node -> graph index in batch
    """

    def __init__(self, node_feat_dim: int, hidden_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(node_feat_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, graph_id: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.lin1(x))
        deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        agg = adj @ h / deg
        h = F.relu(self.lin2(agg))

        num_graphs = int(graph_id.max().item()) + 1
        out = torch.zeros(num_graphs, h.size(-1), device=h.device, dtype=h.dtype)
        out.index_add_(0, graph_id, h)

        count = torch.bincount(graph_id, minlength=num_graphs).float().unsqueeze(-1).to(h.device)
        return out / count.clamp_min(1.0)


class GraphActorCritic(nn.Module):
    def __init__(self, encoder: nn.Module, emb_dim: int, action_dim: int):
        super().__init__()
        self.encoder = encoder
        self.policy_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, 1),
        )
        # Cost value for constrained PPO (estimate expected constraint cost)
        self.cost_value_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, 1),
        )

    def forward(self, graph_state: Dict[str, torch.Tensor]) -> Tuple[Categorical, torch.Tensor, torch.Tensor]:
        z = self.encoder(graph_state["x"], graph_state["adj"], graph_state["graph_id"])
        logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)
        cost_value = self.cost_value_head(z).squeeze(-1)
        dist = Categorical(logits=logits)
        return dist, value, cost_value

    @torch.no_grad()
    def act(self, graph_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        dist, value, cost_value = self(graph_state)
        action = dist.sample()
        return {
            "action": action,
            "logp": dist.log_prob(action),
            "value": value,
            "cost_value": cost_value,
        }


class LagrangeMultiplier(nn.Module):
    """
    Lambda >= 0 by softplus reparameterization.
    Supports multiple constraints: lambda shape [num_constraints].
    """

    def __init__(self, num_constraints: int = 1, init_value: float = 0.1):
        super().__init__()
        raw = torch.log(torch.exp(torch.tensor(init_value)) - 1.0)
        self.raw_lambda = nn.Parameter(raw.repeat(num_constraints))

    def value(self) -> torch.Tensor:
        return F.softplus(self.raw_lambda)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    T = rewards.size(0)
    adv = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_value)
    for t in reversed(range(T)):
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        adv[t] = last_gae
        next_value = values[t]
    returns = adv + values
    return adv, returns


class ConstrainedPPOAgent:
    """
    PPO with Lagrangian constraints:
      maximize J_r(pi) subject to J_c_i(pi) <= d_i
    equivalent saddle objective:
      max_pi min_lambda>=0  J_r(pi) - sum_i lambda_i * (J_c_i(pi) - d_i)
    """

    def __init__(
        self,
        model: GraphActorCritic,
        lagrange: LagrangeMultiplier,
        config: PPOConfig,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.lagrange = lagrange.to(device)
        self.cfg = config
        self.device = device
        self.opt_policy = torch.optim.Adam(self.model.parameters(), lr=config.lr_policy)
        self.opt_lambda = torch.optim.Adam(self.lagrange.parameters(), lr=config.lr_lambda)

    def _policy_loss(
        self,
        ratio: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
        entropy: torch.Tensor,
    ) -> torch.Tensor:
        lam = self.lagrange.value().detach()[0]
        merged_adv = adv_r - lam * adv_c

        surr1 = ratio * merged_adv
        surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * merged_adv
        return -(torch.min(surr1, surr2).mean() + self.cfg.entropy_coef * entropy.mean())

    def _lambda_loss(self, ep_cost_mean: torch.Tensor, cost_limit: float) -> torch.Tensor:
        # Minimize wrt lambda: -lambda * (J_c - d), then lambda increases when cost > limit.
        lam = self.lagrange.value()
        violation = ep_cost_mean.detach() - cost_limit
        return -(lam * violation).mean()

    def update(self, batch: Dict[str, torch.Tensor], cost_limit: float) -> Dict[str, float]:
        # Required batch keys:
        # old_logp, action, value, cost_value, reward, cost, done,
        # new_logp, new_value, new_cost_value, entropy
        old_logp = batch["old_logp"].to(self.device)
        action = batch["action"].to(self.device)
        old_value = batch["value"].to(self.device)
        old_cost_value = batch["cost_value"].to(self.device)
        reward = batch["reward"].to(self.device)
        cost = batch["cost"].to(self.device)
        done = batch["done"].to(self.device)

        with torch.no_grad():
            adv_r, ret_r = compute_gae(
                reward,
                old_value,
                done,
                next_value=old_value[-1],
                gamma=self.cfg.gamma,
                gae_lambda=self.cfg.gae_lambda,
            )
            adv_c, ret_c = compute_gae(
                cost,
                old_cost_value,
                done,
                next_value=old_cost_value[-1],
                gamma=self.cfg.gamma,
                gae_lambda=self.cfg.gae_lambda,
            )

            adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
            adv_c = (adv_c - adv_c.mean()) / (adv_c.std() + 1e-8)

        metrics = {}
        for _ in range(self.cfg.ppo_epochs):
            new_logp = batch["new_logp"].to(self.device)
            new_value = batch["new_value"].to(self.device)
            new_cost_value = batch["new_cost_value"].to(self.device)
            entropy = batch["entropy"].to(self.device)

            ratio = torch.exp(new_logp - old_logp)
            pi_loss = self._policy_loss(ratio, adv_r, adv_c, entropy)

            v_loss = F.mse_loss(new_value, ret_r)
            vc_loss = F.mse_loss(new_cost_value, ret_c)
            total_loss = pi_loss + self.cfg.value_coef * (v_loss + vc_loss)

            self.opt_policy.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
            self.opt_policy.step()

            with torch.no_grad():
                approx_kl = (old_logp - new_logp).mean().item()
            if approx_kl > 1.5 * self.cfg.target_kl:
                break

        ep_cost_mean = cost.mean()
        lambda_loss = self._lambda_loss(ep_cost_mean, cost_limit)
        self.opt_lambda.zero_grad()
        lambda_loss.backward()
        self.opt_lambda.step()

        metrics["pi_loss"] = float(pi_loss.detach().cpu())
        metrics["v_loss"] = float(v_loss.detach().cpu())
        metrics["vc_loss"] = float(vc_loss.detach().cpu())
        metrics["lambda"] = float(self.lagrange.value().detach().cpu()[0])
        metrics["ep_cost"] = float(ep_cost_mean.detach().cpu())
        metrics["approx_kl"] = float(approx_kl)
        return metrics
