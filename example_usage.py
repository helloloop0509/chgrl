import torch

from constrained_ppo_gnn import (
    ConstrainedPPOAgent,
    GraphActorCritic,
    LagrangeMultiplier,
    PPOConfig,
    SimpleGraphEncoder,
)


def fake_graph_batch(
    num_graphs: int = 8,
    nodes_per_graph: int = 5,
    feat_dim: int = 16,
    device: torch.device | str = "cpu",
):
    n = num_graphs * nodes_per_graph
    x = torch.randn(n, feat_dim, device=device)
    adj = torch.eye(n, device=device)
    graph_id = torch.arange(num_graphs, device=device).repeat_interleave(nodes_per_graph)
    return {"x": x, "adj": adj, "graph_id": graph_id}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Please run on a GPU node and re-activate chgrl_venv.")
    device = "cuda"

    cfg = PPOConfig()
    # Demo mode: batch stores one forward-pass graph, so use one PPO epoch.
    # In full training, recompute policy/value each epoch/minibatch instead.
    cfg.ppo_epochs = 1
    encoder = SimpleGraphEncoder(node_feat_dim=16, hidden_dim=64)
    model = GraphActorCritic(encoder=encoder, emb_dim=64, action_dim=6)
    lagrange = LagrangeMultiplier(num_constraints=1, init_value=0.1)
    agent = ConstrainedPPOAgent(model=model, lagrange=lagrange, config=cfg, device=device)

    state = fake_graph_batch(device=device)
    # Old policy/value snapshots from rollout (no grad graph).
    with torch.no_grad():
        old_dist, old_value, old_cost_value = model(state)
        action = old_dist.sample()
        old_logp = old_dist.log_prob(action)

    # Current policy/value for PPO update (with grad graph).
    new_dist, new_value, new_cost_value = model(state)
    new_logp = new_dist.log_prob(action)
    entropy = new_dist.entropy()

    # Demo batch. In real training, these come from rollout trajectories.
    T = action.shape[0]
    batch = {
        "old_logp": old_logp,
        "action": action,
        "value": old_value,
        "cost_value": old_cost_value,
        "reward": torch.randn(T, device=device),
        "cost": torch.rand(T, device=device),  # raw constraint cost, not mixed into reward
        "done": torch.zeros(T, device=device),
        "new_logp": new_logp,
        "new_value": new_value,
        "new_cost_value": new_cost_value,
        "entropy": entropy,
    }

    metrics = agent.update(batch=batch, cost_limit=0.2)
    print(metrics)


if __name__ == "__main__":
    main()
