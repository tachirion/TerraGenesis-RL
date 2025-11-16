# TerraGenesis RL — Report

## Problem definition
(Describe the objective: use continuous control RL to maximize planet habitability while respecting resource budget and stability constraints.)

## Environment
- State space: 5-dim vector `[temp, oxy, pressure, water, habitability_proxy]`
- Action space: 4-dim continuous control in `[-1,1]`
- Episode length: up to 200 steps
- Reward: dense reward composed of task objective (squared error to target), stability penalty, action cost; termination on stability or resource exhaustion.

## Algorithms
We compare: SAC, TD3, DDPG, PPO (stable-baselines3 implementations).
# TODO
- brief algorithm summaries (1–2 paragraphs each)

## Training setup
- Seeds: 42 (global randomness controlled)
- Parallel envs: 4 (DummyVecEnv)
- Timesteps: 300k per algorithm (recommend increasing to 1M for final experiments)
- Hyperparameters: (paste from `code/config.yaml`)

## Results
- Learning curves: returns (plots/learning_returns.png)
- Evaluation: greedy (deterministic) policies evaluated for 10 episodes (CSV in logs/eval). Include a table with mean return, std, mean habitability.

## Observations & analysis
- Which algorithm converged fastest, which achieved the highest habitability, stability vs resource cost trade-offs.

## Challenges & future work
- tuning reward scaling
- using larger batch sizes or more envs
- curriculum learning or shaped reward
- safety constraints and multi-objective optimization


Convert to PDF with `pandoc report.md -o report.pdf` or print/save from your editor.