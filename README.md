# RL Project — TerraGenesis
Made by Tatev Stepanyan and Levon Gevorgyan

## Environment
`code/envs.py` contains the Gymnasium environment `TerraGenesisEnv`.

## Install
Create a python venv and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training

Train all configured algorithms:

```bash
cd TerraGenesis_RL_Project
python code/train.py --config code/config.yaml
```

Adjust timesteps and n_envs in code/config.yaml. TensorBoard logs saved under logs/tensorboard/.

To train a single algorithm only, edit `config.yaml` or modify `train.py` to select a subset.

## Evaluation

Evaluate a trained model (greedy, deterministic):

```bash
python code/evaluate.py --config code/config.yaml --algo SAC --model models/SAC/SAC_final --n_episodes 10
```

Replace SAC and model path with other algorithms.

## Plots

After training, run:

```bash
python code/plots.py
```

This will create learning curves under plots/.

## Deliverables

`code/` — training and evaluation scripts

`models/` — saved checkpoints (created by training)

`logs/` — tensorboard (logs/tensorboard) and CSV logs (logs/csv)

`plots/` — learning curves & evaluation figures

`report/report.md` — report