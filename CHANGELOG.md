# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

### Added
- Initial constrained PPO + GNN state baseline
- Lagrange multiplier based constraint optimization
- GPU demo usage script for Nibi
- Basic open-source project scaffolding

### Changed
- Example training flow adjusted for single-graph demo stability

### Fixed
- Gradient graph issues in demo update path

## [0.1.0] - 2026-03-27

### Added
- `constrained_ppo_gnn.py` core implementation
- `example_usage.py` and `test.py` sanity scripts
- `README.md`, `LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`
- Dependency and tooling configuration (`requirements*.txt`, `pyproject.toml`)
