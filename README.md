# ACAT


## Installation 

Locally:

```
conda create -n acat python=3.6
conda activate acat
pip install -r requirements.txt
python setup.py develop
wandb login
```

## Usage

```
python acat/runner.py with h.gpus=0 h.model_opt.lr=0.01
```


## Debugging

```
OMP_NUM_THREADS=1 python acat/runner.py with h.gpus=0 h.model=linear
WANDB_MODE=dryrun python ...
```

- Use `OMP_NUM_THREADS` to avoid warning on local machine while testing
- Use `WANDB_MODE=dryrun` to not log to wandb
