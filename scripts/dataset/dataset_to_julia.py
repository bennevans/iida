
import pickle
from pandas import DataFrame
import h5py

def unNumpy(list_of_arrays):
    new_list = []
    for l in list_of_arrays:
        new_list.append(l.tolist())
    return new_list


if __name__ == '__main__':
    in_location = '/data/mob/push_box_se_only_offset_K_100_R_20_seed_0.pickle'
    out_location = '/data/mob/push_box_se_only_offset_K_100_R_20_seed_0.h5'

    with open(in_location, 'rb') as f:
        dataset = pickle.load(f)

    groups = ['state', 'act', 'fov']

    hf = h5py.File(out_location, 'w')

    for group in groups:
        groupf = hf.create_group(group)
        for i, data in enumerate(dataset[group]):
            groupf.create_dataset(str(i), data=data)
    
    hf.close()

    # dataset['obs'] = unNumpy(dataset['obs'])
    # dataset['act'] = unNumpy(dataset['act'])
    # dataset['fov'] = unNumpy(dataset['fov'])
    # dataset['rew'] = unNumpy(dataset['rew'])
    # dataset['is_start'] = unNumpy(dataset['is_start'])

    # df = DataFrame(dataset)
    # feather.write_dataframe(df, out_location)