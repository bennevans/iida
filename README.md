# varyingsim

# Installation Instructions
Clone the repo
```
git clone git@github.com:bennevans/varyingsim.git
cd varyingsim/
```

Run one of 
```
conda env create -f setup/environment.yaml
```
OR 
```
cd setup/
pip install -r requirements.txt
```

Run 
```
pip install -e . 
```

# Dataset Generation Instructions

# Run instructions
To run with the current experiment and algorithm parameters:
```
cd scripts/
```
And run:
```
python run_exp.py -c <config file>.yaml
```

## Experiment parameters
