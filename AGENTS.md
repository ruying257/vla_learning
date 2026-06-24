# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python robotics learning project built around MuJoCo, LeRobot, and ACT policy training. The top-level numbered scripts define the main workflow: `1.collect_data.py`, `2.visualize_data.py`, `3.train.py`, `4.deploy.py`, and `4.deploy_headless.py`. Core simulation and robot utilities live in `mujoco_env/`. MuJoCo XML models, meshes, and object assets live in `mode/`; visual README assets live in `assets/`; datasets and generated runs are under `datasets/`, `ckpt/`, `videos/`, and `exp_log/`. Experiment orchestration and summaries live in `scripts/`, with operating notes in `docs/`.

## Build, Test, and Development Commands

- `pip install -r requirements.txt`: install the pinned runtime dependencies.
- `python -m py_compile 1.collect_data.py 2.visualize_data.py 3.train.py 4.deploy.py 4.deploy_headless.py`: quick syntax check for the primary entry points.
- `python 1.collect_data.py`: collect keyboard teleoperation demonstrations.
- `python 2.visualize_data.py`: replay collected demonstrations.
- `python 3.train.py`: train the ACT policy and write checkpoints.
- `python 4.deploy.py`: run policy deployment with viewer support.
- `ACT_RECORD_VIDEO=1 ACT_DEPLOY_MAX_STEPS=300 python 4.deploy_headless.py`: run headless deployment for server or WSL usage.

## Coding Style & Naming Conventions

Use Python 3.10+ and follow the existing plain-script style. Keep 4-space indentation, snake_case for functions and variables, PascalCase for classes, and uppercase names for constants. Prefer small, local changes over broad rewrites. When adding code comments, make them useful and write them in Chinese. Keep runtime paths explicit and avoid hard-coded machine-specific absolute paths.

## Testing Guidelines

There is no formal test suite in the repository. Before handing off changes, run `python -m py_compile` on touched Python files. For behavior changes, run the smallest relevant workflow command, such as a short visualization, a smoke training run using environment overrides, or a bounded headless deployment. Document any command that cannot be run because it requires GPU, display access, or existing checkpoints.

## Commit & Pull Request Guidelines

Recent history uses a mix of Conventional Commit style, such as `docs:` and `chore(scripts/docs):`, plus concise Chinese summaries. Prefer `type(scope): summary` when practical, and keep the summary focused on the observable change. Pull requests should describe the goal, list verification commands and results, mention affected datasets/checkpoints/configuration, and include screenshots or video links when viewer or deployment output changes.

## Agent-Specific Instructions

先给结论，再给依据。解释概念时优先用第一性原理。新增代码时加适当中文注释，并在行为或入口变化时同步更新对应 README 或 `docs/` 文档。
