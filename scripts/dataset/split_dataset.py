from varyingsim.datasets.fov_dataset import split_episodic
import pickle
import os

base_dir = '/data/varyingsim/datasets/'

location = os.path.join(base_dir, 'push_box_se_same_act_K_100_R_2000_seed_0.pickle')

with open(location, 'rb') as f:
    data = pickle.load(f)

K_train, R_train = 90, 1000
K, R = data['K'], data['R']
seed = 0

train_data, test_state_data, test_fov_data, test_fov_state_data = split_episodic(location, K_train, R_train)

names = [   'push_box_se_same_act_split_train_K_{}_R_{}_KT_{}_RT_{}_seed_{}.pickle'.format(K, R, K_train, R_train, seed),
            'push_box_se_same_act_split_test_state_K_{}_R_{}_KT_{}_RT_{}_seed_{}.pickle'.format(K, R, K_train, R_train, seed),
            'push_box_se_same_act_split_test_fov_K_{}_R_{}_KT_{}_RT_{}_seed_{}.pickle'.format(K, R, K_train, R_train, seed),
            'push_box_se_same_act_split_test_fov_state_K_{}_R_{}_KT_{}_RT_{}_seed_{}.pickle'.format(K, R, K_train, R_train, seed),
        ]

datas = [train_data, test_state_data, test_fov_data, test_fov_state_data]

assert len(names) == len(datas)

for name, data in zip(names, datas):
    write_loc = os.path.join(base_dir, name)
    with open(write_loc, 'wb') as f:
        pickle.dump(data, f)
