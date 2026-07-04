# AGENTS.md

## Cursor Cloud specific instructions

LeRobot is a Python (>=3.12) robotics ML library/CLI — there is no long-running
server or web app. The "application" is the set of `lerobot-*` CLI entrypoints
(training, evaluation, dataset tools, robot control). See `README.md` and
`docs/source/installation.mdx` for the canonical usage docs.

### Environment

- Dependencies live in a `uv`-managed virtualenv at `.venv` in the repo root
  (created/refreshed by the startup update script). Activate with
  `source .venv/bin/activate`, or invoke tools directly via `.venv/bin/<tool>`.
- This VM is **CPU-only** (no CUDA/GPU). Always pass `--policy.device=cpu` to
  training/eval commands.
- Installed extras: `test`, `dev`, `aloha`, `pusht`. Real-robot hardware extras
  (feetech/dynamixel/etc.) and heavy policy extras (`smolvla`, `pi`, `groot`,
  `hilserl`, `xvla`, ...) are NOT installed by default — add them with
  `uv pip install -e ".[<extra>]"` if a task needs them.

### Makefile gotcha (important)

The `Makefile` computes `PYTHON_PATH := $(shell .venv/bin/python)`, which
**executes `python` with no arguments**. When `make` runs under a TTY (e.g. a
tmux/interactive shell), this launches an interactive Python REPL and `make`
hangs forever. To run the end-to-end `test-*-ete-*` targets, either:

- run the underlying `lerobot-train` / `lerobot-eval` command directly (preferred), or
- redirect stdin, e.g. `make test-act-ete-train < /dev/null`.

### Lint / test / run

- Lint: the repo uses `pre-commit` (ruff, ruff-format, typos, bandit, mypy;
  config in `pyproject.toml`). For a quick check without the full hook install:
  `.venv/bin/python -m ruff check src/lerobot` and
  `ruff format --check src/lerobot`.
- Tests: `pytest tests/<path>`. Test fixtures under `tests/artifacts/` are
  stored in **git-lfs**; the update script assumes they are already fetched
  (`git lfs pull`). Some tests require extra extras or hardware and will skip.
- Core hello-world flow (train + evaluate a policy in simulation) downloads
  small public datasets from the Hugging Face Hub. Example (CPU):
  `lerobot-train --policy.type=act --policy.device=cpu --env.type=aloha \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human --steps=4 \
  --batch_size=2 --wandb.enable=false --output_dir=tests/outputs/act/`
  then evaluate the saved checkpoint with `lerobot-eval --policy.path=<ckpt> \
  --policy.device=cpu --env.type=aloha`.
