# Contributing

Thanks for considering contributing to CHGRL.

## Development Setup

1. Create/activate environment:

```bash
module load python/3.10 cuda/12.6
source ~/chgrl_venv/bin/activate
```

2. Install dev dependencies:

```bash
pip install -r requirements-dev.txt
```

## Code Style

- Format with `black`
- Sort imports with `isort`
- Lint with `ruff`
- Keep code Python 3.10 compatible

Example:

```bash
black .
isort .
ruff check .
```

## Pull Requests

- Keep PRs focused and small
- Include a clear description and motivation
- Add/update docs when behavior changes
- Include a minimal test or reproduction script
