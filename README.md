# Context is Everything: Implicit Identification for Dynamics Adaptation

[Project Page](https://bennevans.github.io/iida/)

## Installation Instructions
1. Clone the repo
```
git clone git@github.com:bennevans/iida.git
cd iida/
```

2. Install MuJoCo [here](https://mujoco.org/download).

3. Install the required python packages:
```
conda env create -f setup/environment.yaml
```
OR 
```
cd setup/
pip install -r requirements.txt
```

3. Install the `varyingsim` library
```
pip install -e . 
```

4. Generate the simulated dataset or download it [here](todo.com). The robot dataset can be downloaded [here](todo.com).

## Dataset Generation Instructions
### Single-step Environments
```
cd scripts/dataset/
python create_da_dataset_push_box.py
python create_da_dataset_slide_puck.py
```
### Multi-step Environments
To train a policy to generate rollouts, I use [mjrl](https://github.com/aravindr93/mjrl).
```
cd scripts/policy/
python train_policy.py
python relabel_paths.py --env <env>
```
## Run instructions
To run with the current experiment and algorithm parameters:
```
cd scripts/
```
And run:
```
python run_exp.py -c <config file>.yaml
```
