
import unittest

from varyingsim.datasets.fov_dataset import SmoothFovDataset

class DatasetTest(unittest.TestCase):

    location = '/data/varyingsim/push_box_contact_simple_large_1.pickle'
    H = 8
    dataset = SmoothFovDataset(location, H, obs_skip=1)
    dataset_2 = SmoothFovDataset(location, H, obs_skip=2)
    dataset_20 = SmoothFovDataset(location, H, obs_skip=20)

    def test_obs(self):
        datum = self.dataset[0]
        datum_2 = self.dataset_2[0]
        datum_20 = self.dataset_20[0]

        self.assertTrue((datum['obs'] == datum_2['obs']).all())
        self.assertTrue((datum['obs'] == datum_20['obs']).all())

        datum = self.dataset[20]
        datum_2 = self.dataset_2[10]
        datum_20 = self.dataset_20[1]

        self.assertTrue((datum['obs'] == datum_2['obs']).all())
        self.assertTrue((datum['obs'] == datum_20['obs']).all())

        datum = self.dataset[self.dataset.prefix_lens[1]]
        datum_2 = self.dataset_2[self.dataset_2.prefix_lens[1]]
        datum_20 = self.dataset_20[self.dataset_20.prefix_lens[1]]

        self.assertTrue((datum['obs'] == datum_2['obs']).all())
        self.assertTrue((datum['obs'] == datum_20['obs']).all())

        datum = self.dataset[self.dataset.prefix_lens[1] + 40]
        datum_2 = self.dataset_2[self.dataset_2.prefix_lens[1] + 20]
        datum_20 = self.dataset_20[self.dataset_20.prefix_lens[1] + 2]

        self.assertTrue((datum['obs'] == datum_2['obs']).all())
        self.assertTrue((datum['obs'] == datum_20['obs']).all())

    def test_context_obs_self(self):
        # Test that first trajectory's context_obs is consistient with the obs
        datum_0 = self.dataset_20[0]
        datum_1 = self.dataset_20[1]
        datum_2 = self.dataset_20[2]
        datum_3 = self.dataset_20[3]
        
        self.assertTrue((datum_1['context_obs'][0] == datum_0['obs']).all())

        self.assertTrue((datum_2['context_obs'][0] == datum_0['obs']).all())
        self.assertTrue((datum_2['context_obs'][1] == datum_1['obs']).all())

        self.assertTrue((datum_3['context_obs'][0] == datum_0['obs']).all())
        self.assertTrue((datum_3['context_obs'][1] == datum_1['obs']).all())
        self.assertTrue((datum_3['context_obs'][2] == datum_2['obs']).all())


        # Test that a different trajectory's context_obs is consistient with the obs
        diff_traj_idx = self.dataset_20.prefix_lens[10]

        datum_0 = self.dataset_20[diff_traj_idx + 5]
        datum_1 = self.dataset_20[diff_traj_idx + 6]
        datum_2 = self.dataset_20[diff_traj_idx + 7]
        datum_3 = self.dataset_20[diff_traj_idx + 8]
        
        self.assertTrue((datum_1['context_obs'][-1] == datum_0['obs']).all())

        self.assertTrue((datum_2['context_obs'][-2] == datum_0['obs']).all())
        self.assertTrue((datum_2['context_obs'][-1] == datum_1['obs']).all())

        self.assertTrue((datum_3['context_obs'][-3] == datum_0['obs']).all())
        self.assertTrue((datum_3['context_obs'][-2] == datum_1['obs']).all())
        self.assertTrue((datum_3['context_obs'][-1] == datum_2['obs']).all())


    def test_context_obs_other(self):

        # consistiency between datasets
        datum = self.dataset[8]
        datum_2 = self.dataset_2[4]

        self.assertTrue((datum['context_obs'][0] == datum_2['context_obs'][0]).all())
        self.assertTrue((datum['context_obs'][2] == datum_2['context_obs'][1]).all())
        self.assertTrue((datum['context_obs'][4] == datum_2['context_obs'][2]).all())
        self.assertTrue((datum['context_obs'][6] == datum_2['context_obs'][3]).all())

        datum = self.dataset[8]
        datum_20 = self.dataset_20[4]
        datum_202 = self.dataset_20[5]

        self.assertTrue((datum['context_obs'][0] == datum_20['context_obs'][0]).all())
        self.assertTrue((datum_20['context_obs'][-1] == datum_202['context_obs'][-2]).all())

        datum = self.dataset[self.dataset.prefix_lens[1] + 40]
        datum_2 = self.dataset_2[self.dataset_2.prefix_lens[1] + 20]
        datum_20 = self.dataset_20[self.dataset_20.prefix_lens[1] + 3]

        self.assertTrue((datum['context_obs'][-2] == datum_2['context_obs'][-1]).all())
        self.assertTrue((datum_20['context_obs'][0] == self.dataset[self.dataset.prefix_lens[1]]['obs']).all())
        self.assertTrue((datum_20['context_obs'][1] == self.dataset[self.dataset.prefix_lens[1] + 20]['obs']).all())
        self.assertTrue((datum_20['context_obs'][2] == self.dataset[self.dataset.prefix_lens[1] + 2*20]['obs']).all())


if __name__ == '__main__':
    unittest.main()
