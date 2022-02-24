

import argparse


from varyingsim.util.parsers import load_everything

if __name__ == '__main__':
    model, train_dataset, test_sets, val_set = load_everything('/data/domain_adaptation/experiments/sliding/baseline/sliding_transformer-2022-01-11_18-49-46')
