import wandb
import numpy as np
import pandas as pd

api = wandb.Api()

# take in run names, get best errors and their confidence intervals
project_name = 'domain_adaptation_icra'

# slide puck
slide_puck_info = {
    'name': 'slidepuck',
    'key_names': ['test-ribbon_base_loss', 'test-train-seed_base_loss', 'test-test-seed_base_loss', 'test-noise_base_loss'],
    'best_key': 'test-noise_base_loss',
    'runs':[
        ['941nssph', 'b03iauv8', 'apano2z4'], # continuous
        ['19hxqyc3', '31ovejeb', '2w13nqpl'], # fov
        ['3osexvyn', '3l99645y', '200lwa0g'], # ff
        ['3gzrl2fo', '1t03es5o', '2r2g80qu'], # rnn
        ['1xe7l6xt', '18indyvm', 'h3nb2ucj'], # transformer
        ['9qdwos0r', '3i9qq3fy', '3avm9ugx'], # single model
        ['15xqzrts', 'sh01x161', '3c0fcr60'], # maml
    ],
    'keys':['continuous', 'fov', 'ff', 'rnn', 'transformer', 'single', 'maml']
}
# push box
push_box_info = {
    'name': 'pushbox',
    'key_names': ['test-state_base_loss', 'test_base_loss', 'test-fov-state_base_loss', 'test-fov_base_loss'],
    'best_key': 'test_base_loss',
    'runs': [
        ['rq3q5x9y', '3ebiongf', '20580owd'], # continuous
        ['34mt27p2', 'tjhcwtas', 'dyk41hoj'], # fov
        ['1dkxnv9q', 'o6ryqd93', '205d4350'], # ff
        ['3827z49a', '29xgduwi', '1wcm11of'], # rnn
        ['1jf9wkfj', '3ae7owfz', '1t4g9r4s'], # transformer
        ['2ucyv0vy', '1zx822lj', '36ad0btl'], # single model
        ['1qyya2ed', '3hei2krz', '3crzh4c0'], # maml
    ],
    'keys':['continuous', 'fov', 'ff', 'rnn', 'transformer', 'single', 'maml']
}

# BAD PUSHBOX!!!
# ['1cbp7ray', '3fy4ztxh', 'kcgduh17'], # continuous
# ['jp2mx6c5', 'i87dkhxn', 'k7kcs3x0'], # fov
# ['tnm4yi6t', 'dajc2fjm', '1dwbog97'], # ff
# ['3goo5d8c', '2fx1w1gz', '1sscfr0q'], # rnn
# ['3fm2rweq', '3fhvmp5y', '3718emy2'], # transformer
# ['asdf', 'qwerty', 'poiuy'], # single model
# ['2tbg51r0', '15qpc9u1', '267y67l6'], # maml

# # hopper
# ['35sxoepo', '1nk9qdgv', '2ktydf21'], # continuous
# ['1jmlvi1m', '3fiojsup', 'hq3obozr'], # fov
# ['270x46ui', 'ozpaxxk7', '2rjo1ixa'], # ff
# ['5oix54qw', '39tk4suk', '18coj1jf'], # rnn
# ['qzyva1y9', '1vm1q1r3', '21h7vi8s'], # transformer
# ['11btjc5p', '2dpkvqlk', '4n953r37'], # single model
# ['asdf', 'qwerty', 'poiuy'], # maml DIDN'T WORK

hopper_info = {
    'name': 'hopper',
    'key_names': ['test-ribbon_base_loss'],
    'best_key': 'test-ribbon_base_loss',
    'runs':[
        ['35sxoepo', '1nk9qdgv', '2ktydf21'], # continuous
        ['1jmlvi1m', '3fiojsup', 'hq3obozr'], # fov
        ['270x46ui', 'ozpaxxk7', '2rjo1ixa'], # ff
        ['5oix54qw', '39tk4suk', '18coj1jf'], # rnn
        ['qzyva1y9', '1vm1q1r3', '21h7vi8s'], # transformer
        ['11btjc5p', '2dpkvqlk', '4n953r37'], # single model
        ['asdf', 'qwerty', 'poiuy'], # maml DIDN'T WORK
    ],
    'keys':['continuous', 'fov', 'ff', 'rnn', 'transformer', 'single', 'maml']
}

# # swimmer
# ['1e8o93ru', '1jnubpz4', '30jb0c1m'], # continuous
# ['zwxemebb', '12x9v5xk', '15ddf0ji'], # fov
# ['eeosv26o', '11kn9r1s', 'zm7g4ccg'], # ff
# ['2oii46un', '3u2eslzh', '3ctnb4gu'], # rnn
# ['2ecszar4', '18j5275o', '2dbmvhf1'], # transformer
# ['1gldrh9v', '2ex4zsz4', '3l7fixqn'], # single model
# ['asdf', 'qwerty', 'poiuy'], # maml DIDN'T WORK

swimmer_info = {
    'name': 'swimmer',
    'key_names': ['test-ribbon_base_loss'],
    'best_key': 'test-ribbon_base_loss',
    'runs':[
        ['1e8o93ru', '1jnubpz4', '30jb0c1m'], # continuous
        ['zwxemebb', '12x9v5xk', '15ddf0ji'], # fov
        ['eeosv26o', '11kn9r1s', 'zm7g4ccg'], # ff
        ['2oii46un', '3u2eslzh', '3ctnb4gu'], # rnn
        ['2ecszar4', '18j5275o', '2dbmvhf1'], # transformer
        ['1gldrh9v', '2ex4zsz4', '3l7fixqn'], # single model
        ['asdf', 'qwerty', 'poiuy'], # maml DIDN'T WORK
    ],
    'keys':['continuous', 'fov', 'ff', 'rnn', 'transformer', 'single', 'maml']
}

# # humanoid
# ['6sjo1a0m', '1i9y0dnu', '17u5ooec'], # continuous
# ['2m9cf5yp', '1wwndkm9', '1zt3hm71'], # fov
# ['1fbiff8j', '3ivq6d42', '435cvn03'], # ff
# ['1drt2z7t', '2y5wvj2e', '35q4kwrw'], # rnn
# ['m1jnch3w', '2pw8om19', '29uhsbmy'], # transformer
# ['13fgywpd', '3npk9mp3', '1kobclxh'], # single model
# ['asdf', 'qwerty', 'poiuy'], # maml DIDN'T WORK

humanoid_info = {
    'name': 'humanoid',
    'key_names': ['test-ribbon_base_loss', 'test-ribbon-3_base_loss'],
    'best_key': 'test-ribbon_base_loss',
    'runs':[
        ['6sjo1a0m', '1i9y0dnu', '17u5ooec'], # continuous
        ['2m9cf5yp', '1wwndkm9', '1zt3hm71'], # fov
        ['1fbiff8j', '3ivq6d42', '435cvn03'], # ff
        ['1drt2z7t', '2y5wvj2e', '35q4kwrw'], # rnn
        ['m1jnch3w', '2pw8om19', '29uhsbmy'], # transformer
        ['13fgywpd', '3npk9mp3', '1kobclxh'], # single model
        ['asdf', 'qwerty', 'poiuy'], # maml DIDN'T WORK
    ],
    'keys':['continuous', 'fov', 'ff', 'rnn', 'transformer', 'single', 'maml']
}

# robot
robot_info = {
    'name': 'sliding',
    'key_names': ['test_base_loss'],
    'best_key': 'test_base_loss',
    'runs':[
        ['140ljpfa', '1yt3ht83', '315qnjfq'], # continuous
        ['asdf', 'qwerty', 'poiuy'], # fov
        ['44j73dn7', '27reywb5', '2x6fhddq'], # ff
        ['2p0y7a95', '1is72e8f', 'djfzgc7k'], # rnn
        ['738qryls', 'jtwpiq87', '23qrl6u0'], # transformer
        ['25t1rlhb', 'r98zgz98', '278touo7'], # single model
        ['asdf', 'qwerty', 'poiuy'], # maml
    ],
    'keys':['continuous', 'fov', 'ff', 'rnn', 'transformer', 'single', 'maml']
}

# # to copy
# # ['asdf', 'qwerty', 'poiuy'], # continuous
# # ['asdf', 'qwerty', 'poiuy'], # fov
# # ['asdf', 'qwerty', 'poiuy'], # ff
# # ['asdf', 'qwerty', 'poiuy'], # rnn
# # ['asdf', 'qwerty', 'poiuy'], # transformer
# # ['asdf', 'qwerty', 'poiuy'], # single model
# # ['asdf', 'qwerty', 'poiuy'], # maml

def get_stats(runs, best_key):
    best_vals = []
    for run_name in runs:
        run_name_full = "{}/{}".format(project_name, run_name)
        run = api.run(run_name_full)
        hist = run.history()
        best_val = np.min(hist[best_key])
        best_vals.append(best_val)
        
    return np.mean(best_vals), np.std(best_vals)

# infos = [slide_puck_info, push_box_info, hopper_info, swimmer_info, humanoid_info, robot_info]
infos = [robot_info]


mses = pd.DataFrame(columns=['environment'] + humanoid_info['keys'])
stds = pd.DataFrame(columns=['environment'] + humanoid_info['keys'])

for info in infos:
    name = info['name']
    print(name)
    row_mses = {}
    row_stds = {}
    for key, runs in zip(info['keys'], info['runs']):
        try:
            mean, std = get_stats(runs, info['best_key'])
            # mean_str = format(mean, ".2e").replace('e-0', 'e-')
            # std_str = format(std, ".2e").replace('e-0', 'e-')
            # stored_str = "{} ({})".format(mean_str, std_str)
            row_mses[key] = mean
            row_stds[key] = std
        except Exception as e:
            row_mses[key] = np.nan
            row_stds[key] = np.nan

    row_mses['environment'] = name
    row_stds['environment'] = name
    mses = mses.append(row_mses, ignore_index=True)
    stds = stds.append(row_stds, ignore_index=True)


def divide(tens):
    def my_fun(x):
        if type(x) is not str and np.isscalar(x):
            return x / tens
        return x
    return my_fun

out_strs = pd.DataFrame(columns=['environment'] + humanoid_info['keys'])

for (i, mse), (i, std) in zip(mses.iterrows(), stds.iterrows()):
    row_min = np.min(mse.dropna()[1:])
    exp = np.floor(np.log10(row_min))
    tens = 10 ** exp
    my_fun = divide(tens)
    mse_applied = mse.apply(my_fun)
    std_applied = std.apply(my_fun)
    
    row_dict = {}
    for key, val in mse_applied.iteritems():
        if key == 'environment':
            row_dict[key] = val + ' $10^{{{}}}$'.format(int(exp))
        else:
            if np.isnan(val):
                item_str = 'N/A'
            else:
                item_str = "{:.2f} $\pm$ {:.2f}".format(mse_applied[key], std_applied[key])
            row_dict[key] = item_str
    out_strs = out_strs.append(row_dict, ignore_index=True)
# all_data.to_csv('table_robot.csv', index=False)
out_strs.to_csv('table_rob.csv', index=False)
"""

data_all = []
best_idxs = []

for run_name in run_names:
    run_name_full = "{}/{}".format(project_name, run_name)
    run = api.run(run_name_full)
    hist = run.history()
    data = {}
    for key in key_names:
        data[key] = hist[key]
    
    best_idx = np.argmin(data[best_key])
    best_idxs.append(best_idx)
    
    best_vals = {}
    for key in key_names:
        best_val = data[key][best_idx]
        best_vals[key] = best_val
    data_all.append(best_vals)

stats_all = {}

for datum in data_all:
    for k,v in datum.items():
        if k in stats_all:
            stats_all[k].append(datum[k])
        else:
            stats_all[k] = [datum[k]]

for k,v in stats_all.items():
    print(k, np.mean(v), np.std(v))


""" 