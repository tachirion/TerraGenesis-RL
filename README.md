# TerraGenesis RL

Training and evaluation framework for a TerraGenesis-inspired Reinforcement Learning environment using
Stable-Baselines3.

---

## Overview

This repository implements a **TerraGenesis-like environment** for Reinforcement Learning experiments.
The environment simulates planetary parameters such as temperature, oxygen, pressure, water, and biomass, and allows
agents to learn policies to optimize **habitability** over time.

It includes:

- **Custom RL Environment** (`envs.py`)  
- **Training scripts** with SAC, TD3, and DDPG (`train.py`)  
- **Evaluation scripts** (`evaluate.py`)  
- **Visualization utilities** for learning curves (`plots.py`)  
- **Hyperparameter optimization** with Optuna  
- **CSV logging** for per-step and per-episode metrics  

---

## Features

- **Supported algorithms:**  
  - SAC (Soft Actor-Critic)  
  - TD3 (Twin Delayed DDPG)  
  - DDPG (Deep Deterministic Policy Gradient)

- **Environment and reward normalization** via `VecNormalize`  
- **Linear learning rate schedules**  
- **Action noise** (OU) with linear annealing  
- **Evaluation callbacks** with best-model saving  
- **CSV & TensorBoard logging**  
- **Safe handling** of environment seeding and configuration parameters  
- **Support for SAC-specific options:** `ent_coef` (alpha) and `target_entropy` scaling  

---

## Installation

1. Clone this repository:

```bash
git clone https://github.com/tachirion/TerraGenesis-RL.git
cd TerraGenesis-RL
```

2. Create a Python environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install requirements:

```bash
pip install -r requirements.txt
```

---


## Configuration
All settings are stored in `config.yaml`:

- `env`
    - `max_steps`: Maximum steps per episode 
    - `stable_steps_required`: Steps required for "stability" termination 
    - `normalize`: Apply `VecNormalize`? 
    - `norm_reward`: Normalize reward?
- `training`
  - `timesteps`: Total training steps 
  - `n_envs`: Number of vectorized environments 
  - `algorithms`: List of algorithms to train 
  - `save_freq`: Frequency of checkpoint saving 
  - `eval_freq`: Frequency of evaluation 
  - `eval_episodes`: Number of episodes during evaluation

- Algorithm-specific options: `sac`, `td3`, `ddpg` 
  - `batch_size`, `buffer_size`, `gamma`, `tau`, `learning_rate` 
  - SAC: `ent_coef`, `target_entropy_scale` 
  - DDPG/TD3: `noise_sigma`, `noise_type`

---

## Training

```bash
python code/train.py --config code/config.yaml --algo SAC
python code/train.py --config code/config.yaml --algo TD3
python code/train.py --config code/config.yaml --algo DDPG
```

Or to train all algorithms sequentially:

```bash
python code/train.py --config code/config.yaml
```

- TensorBoard logs are saved in the folder specified in `config.yaml`. 
- Episode-level CSV logs and step-level logs are automatically generated. 
- Best model checkpoints are saved during training.

---

## Evaluation

```bash
python code/evaluate.py \
  --config code/config.yaml \
  --algo SAC \
  --model models/sac/best_model \
  --n_episodes 200
```

Outputs:
- Episode-level CSV (`logs/eval`)
- Step-level CSV (`logs/eval`)
- Plots: total return and mean habitability (`plots/`)

---

## Visualization

```bash
python code/plots.py
```

- Generates learning curves for all trained algorithms. 
- Compares performance using rolling mean of episode returns.

## Hyperparameter Optimization

Optuna-based tuning for SAC:

```bash
python code/optimize.py
```

- Optimizes: `learning_rate`, `gamma`, `tau`, `batch_size`, network size, `ent_coef` (alpha), `target_entropy_scale`
- Uses a custom pruning callback to terminate unpromising trials early.

---

## Project Structure

```
├── code/
│   ├── envs.py             # Custom RL environment
│   ├── utils.py            # Helpers for seeding and CSV logging
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   ├── plots.py            # Visualization utilities
│   ├── optimize.py         # Optuna hyperparameter tuning
│   └── config.yaml         # Configuration file
├── models/                 # Saved models
├── logs/                   # TensorBoard and CSV logs
├── plots/                  # Generated plots
├── report.pdf              # Project report
├── requirements.txt
└── README.md
```

---

## Notes

- Ensure consistent seeding for reproducibility (`seed` in `config.yaml`). 
- Reward clipping and normalized states improve RL stability. 
- CSV logging enables detailed analysis of agent behavior per step and episode. 
- Hyperparameter tuning via Optuna allows rapid exploration of SAC parameters for improved performance.

---

## References

- Stable-Baselines3 Documentation (https://stable-baselines3.readthedocs.io/)
- Gymnasium Documentation (https://gymnasium.farama.org/)
- TerraGenesis Game (inspiration for environment design) (https://terragenesis.fandom.com/wiki/Main_Page)
- Optuna Hyperparameter Optimization (https://optuna.org/)

---

## License
This project is released under the MIT License. See `LICENSE` for details

## Credits

Made by Tatev Stepanyan and Levon Gevorgyan. Instructor: Davit Ghazaryan.