# CHGRL: Constrained PPO with GNN State

Minimal starter project for:

- GNN-style state encoding
- PPO policy optimization
- Constraint handling with Lagrange multipliers (not reward-weighted penalty mixing)

This repository provides a compact baseline you can extend into a full research/training pipeline.

## Features

- `GraphActorCritic` with:
  - policy head
  - reward value head
  - cost value head
- Lagrangian constrained objective:
  - maximize reward
  - enforce constraints with learnable `lambda >= 0`
- GPU-first demo script on Nibi cluster

## Repository Structure

- `constrained_ppo_gnn.py`: core model and constrained PPO logic
- `example_usage.py`: minimal GPU demo run
- `test.py`: quick CUDA sanity check

## Quick Start (Nibi / Slurm)

### 1) Allocate an interactive GPU node

```bash
salloc --account=def-bkantarc --partition=gpubase_interac --gres=gpu:h100_1g.10gb:1 --cpus-per-task=2 --mem=16G --time=4:00:00
```

### 2) Load modules and activate environment

```bash
module load python/3.10 cuda/12.6
source ~/chgrl_venv/bin/activate
```

### 3) Verify GPU visibility

```bash
python test.py
```

Expected output:

- `True`
- GPU name (for example, H100 MIG profile)

### 4) Run the demo

```bash
python example_usage.py
```

Example output:

```text
{'pi_loss': ..., 'v_loss': ..., 'vc_loss': ..., 'lambda': ..., 'ep_cost': ..., 'approx_kl': ...}
```

## Constraint Formulation

We solve:

- maximize `J_r(pi)`
- subject to `J_c_i(pi) <= d_i`

with the Lagrangian saddle objective:

- `max_pi min_lambda>=0  J_r(pi) - sum_i lambda_i * (J_c_i(pi) - d_i)`

In code:

- policy advantage is merged as `A_r - lambda * A_c`
- `lambda` is updated from constraint violation (`ep_cost - cost_limit`)
- `lambda` is parameterized with `softplus` to keep it non-negative

## Important Notes

- The current `example_usage.py` sets `ppo_epochs = 1` for demo stability because it reuses one forward graph in the synthetic batch.
- For real training, you should recompute policy/value forward pass per epoch/minibatch and then use larger PPO epochs (such as 5-10).

## How to Adapt to Your Environment

1. Replace `fake_graph_batch(...)` with your real graph-state constructor:
   - node features `x`
   - adjacency `adj` or edge list representation
   - graph batch id `graph_id`
2. Replace demo `reward` and `cost` with rollout data from your env.
3. Keep constraints in `cost` channel, not directly merged into reward.
4. Tune:
   - `cost_limit`
   - `lr_lambda`
   - PPO clipping and entropy coefficients

## Next Steps (Suggested)

- Add full rollout buffer and minibatch sampling
- Recompute `new_logp/new_value/new_cost_value` inside each PPO epoch
- Add logging (TensorBoard or Weights & Biases)
- Add evaluation script with reward/constraint metrics

## License

Add a license file (for example, MIT) before public release.
