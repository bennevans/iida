import configargparse

def add_default_options(p):
    p.add('-o', '--output', default='.', help='output directory')
    p.add('-n', '--name', default='', help='experiment name')
    p.add('-e', '--env', required=True, help='environment', choices=\
        ['PushBoxCircle', 'PushBoxOffset', 'DummyEnv', 'SlidePuck', 'CartPole',
        'Hopper', 'Swimmer', 'Humanoid'])
    p.add('-d', '--train-dataset-location', required=True, help='dataset location')
    p.add('--test-datasets', required=False, type=str, nargs='+', help='test dataset locations')
    p.add('--test-names', required=False, type=str, nargs='+', help='test dataset locations')
    p.add('--val-dataset-location', required=False, type=str, default=None, help='validation dataset location')
    p.add('-t', '--dataset-type', required=True, help='dataset type')
    p.add('--obs-skip', type=int, help='the number of observations to skip if using SmoothFovDataset', default=50)
    p.add('--seed', default=0, help='seed', type=int)
    p.add('-x', '--device', default='cuda', help='torch device')
    p.add('--aim_dir', required=False, type=str, help='aim repository directory')
    p.add('-p', '--print-iter', type=int, default=100)
    p.add('-s', '--save-iter', type=int, default=100)
    p.add('-l', '--learn-iters', type=int, default=1)
    p.add('-m', '--message', type=str)
    p.add('--context-size', type=int, default=1)
    p.add('--model-config', type=str, required=True)

    p.add('-b', '--batch-size', type=int, default=64)
    p.add('-r', '--lr', type=float, default=1e-3)
    p.add('--epochs', type=int, default=1)
    p.add('--include-full', type=bool, default=False)
    p.add('--pad-till', type=int, default=-1)
    p.add('--test-batch-size', type=int, default=512)
    p.add('--val-batch-size', type=int, default=512)
    p.add('--n-val-iter', type=int, default=1)
    p.add('--val-iter', type=int, default=50)
    p.add('--train-ratio', type=float, default=0.9)
    p.add('--test-iter', type=int, default=100)
    p.add('--use-obs-fn', dest='use_obs_fn', action='store_true')
    p.add('--n-batch', type=int, default=1)
    p.add('--lr-step-size', type=int, default=None)
    p.add('--lr-gamma', type=float, default=None)

    p.add('--num-workers', type=int, default=8)
    p.add('--no-wandb', action='store_true')
    p.add('--optim', type=str, default='adam', choices=['adam', 'sgd'])

    p.add('--load-model-location', type=str, default=None)
    p.add('--model-name', type=str, default='model.pickle')
    p.add('--freeze-encoder', type=bool, default=False)
    p.add('--group', type=str, default=None)
    p.add('--same-ctx', type=bool, default=False)

    p.add('--env-nu', type=int, default=1)
    p.add('--env-nq', type=int, default=2)
    p.add('--env-nv', type=int, default=0)
    p.add('--obs_fn', type=str, default=None)

    p.add('--project', type=str, default='domain_adaptation_rss')

    return p

def get_default_parser():
    p = configargparse.ArgParser()
    p.add('-c', '--config', is_config_file=True, help='config file path')
    p = add_default_options(p)
    return p
