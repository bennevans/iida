
import pickle
import numpy as np

# this file takes in a data location and creates a new dataset
orig_dataset = '/data/varyingsim/push_box_contact_simple_large_1.pickle'
with open(orig_dataset, 'rb') as f:
    orig_dataset = pickle.load(f)
    
# the new dataset has the trajectories that are difficult to predict removed
keep_idxs = []
for i, fov in enumerate(orig_dataset['fov']):
    if np.abs(fov[0][0]) >= 0.02:
        keep_idxs.append(i)

new_data = {}
new_data['test_traj_len'] = orig_dataset['test_traj_len']
new_data['dataset_type'] = orig_dataset['dataset_type']
new_data['N'] = len(keep_idxs)
data_keys = ['obs', 'act', 'fov', 'is_start']
for key in data_keys:
    new_data[key] = []

for idx in keep_idxs:
    for key in data_keys:
        new_data[key].append(orig_dataset[key][idx])

nocenter_dataset = '/data/varyingsim/push_box_contact_noccenter_large_1.pickle'
with open(nocenter_dataset, 'wb') as f:
    pickle.dump(new_data, f)


# randomly shuffles a dataset
new_data = {}
new_data['test_traj_len'] = orig_dataset['test_traj_len']
new_data['dataset_type'] = orig_dataset['dataset_type']
new_data['N'] = len(keep_idxs)
data_keys = ['obs', 'act', 'fov', 'is_start']

rand_idx = np.random.permutation(orig_dataset['N'])

for key in data_keys:
    new_data[key] = list(np.array(orig_dataset[key])[rand_idx])

shuffled_dataset = '/data/varyingsim/push_box_contact_shuffled_large_1.pickle'
with open(shuffled_dataset, 'wb') as f:
    pickle.dump(new_data, f)

