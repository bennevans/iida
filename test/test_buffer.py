
import unittest
import numpy as np

from copy import deepcopy
from varyingsim.util.buffers import TrajBuffer

class TrajBufferTest(unittest.TestCase):

    datum_1 = [np.array([1,2.]), np.array(1.5)]
    datum_2 = [np.array([3,4.]), np.array(6.)]
    traj_1 = [np.array([[1.,2],[3,4]]), np.array([1.5, 6.])]
    traj_2 = [np.array([[0, 0.],[0, 0.],[0, 0.]]), np.array([0,0,0.])]

    def testAddSimple(self):
        buffer = TrajBuffer(10, [(2,), ()]) 
        buffer.add_datum(self.datum_1)
        buffer.add_datum(self.datum_2)
        self.assertEqual(len(buffer), 2)
    
    def testAddMax(self):
        MAX_SIZE = 10
        buffer = TrajBuffer(MAX_SIZE, [(2,), ()]) 

        for i in range(MAX_SIZE):
            buffer.add_datum(self.datum_2)

        self.assertEqual(len(buffer), MAX_SIZE)

    def testRidOld(self):
        MAX_SIZE = 10
        buffer = TrajBuffer(MAX_SIZE, [(2,), ()]) 

        buffer.add_datum(self.datum_1)

        for i in range(MAX_SIZE):
            buffer.add_datum(self.datum_2)

        for traj in buffer.trajectories:
            for traj_part, datum_part in zip(traj, self.datum_2):
                self.assertTrue((traj_part == datum_part).all())

    def testCreateTraj(self):
        MAX_SIZE = 10
        shapes = [(2,), ()]
        buffer = TrajBuffer(MAX_SIZE, shapes) 
        buffer.add_datum(self.datum_1)
        buffer.add_datum(self.datum_2)
        buffer.add_datum(self.datum_1)
        buffer.add_datum(self.datum_2)

        self.assertEqual(buffer.n_trajs, 1)
        self.assertEqual(len(buffer), 4)

    def testNewTrajs(self):
        MAX_SIZE = 10
        shapes = [(2,), ()]
        buffer = TrajBuffer(MAX_SIZE, shapes) 
        buffer.add_datum(self.datum_1)
        buffer.add_datum(self.datum_2)
        buffer.add_datum(self.datum_1)
        buffer.add_datum(self.datum_2)

        buffer.add_datum(self.datum_1, True)
        buffer.add_datum(self.datum_2)
        buffer.add_datum(self.datum_1)

        self.assertEqual(buffer.n_trajs, 2)
        self.assertEqual(len(buffer), 7)

    def testAddTraj(self):
        MAX_SIZE = 10
        shapes = [(2,), ()]
        buffer = TrajBuffer(MAX_SIZE, shapes) 
        buffer.add_traj(self.traj_1)

        self.assertEqual(buffer.n_trajs, 1)
        self.assertEqual(len(buffer), 2)
    
    def testAddTrajMany(self):
        MAX_SIZE = 10
        shapes = [(2,), ()]
        buffer = TrajBuffer(MAX_SIZE, shapes) 
        buffer.add_traj(self.traj_1)
        buffer.add_traj(self.traj_2)
        
        self.assertEqual(buffer.n_trajs, 2)
        self.assertEqual(len(buffer), 5)

    def testAddTrajManyMax(self):
        MAX_SIZE = 9
        shapes = [(2,), ()]
        buffer = TrajBuffer(MAX_SIZE, shapes) 
        buffer.add_traj(self.traj_1)
        buffer.add_traj(self.traj_2)
        buffer.add_traj(self.traj_1)
        buffer.add_traj(self.traj_2)

        self.assertEqual(buffer.n_trajs, 4)
        self.assertEqual(len(buffer), MAX_SIZE)

    def testAddSingleAndTraj(self):
        MAX_SIZE = 32
        shapes = [(2,), ()]
        buffer = TrajBuffer(MAX_SIZE, shapes) 
        buffer.add_datum(self.datum_1)
        buffer.add_datum(self.datum_2)
        buffer.add_datum(self.datum_1)
        buffer.add_datum(self.datum_2)

        buffer.add_datum(self.datum_1, True)
        buffer.add_datum(self.datum_2)
        buffer.add_datum(self.datum_1)

        buffer.add_traj(self.traj_2)

        self.assertEqual(buffer.n_trajs, 3)
        self.assertEqual(len(buffer), 10)

    def testAddTrajAndSingle(self):
        MAX_SIZE = 32
        shapes = [(2,), ()]

        buffer = TrajBuffer(MAX_SIZE, shapes) 
        buffer.add_traj(self.traj_2)

        buffer.add_datum(self.datum_1, True)
        buffer.add_datum(self.datum_2)
        buffer.add_datum(self.datum_1)
        buffer.add_datum(self.datum_2)

        buffer.add_datum(self.datum_1, True)
        buffer.add_datum(self.datum_2)
        buffer.add_datum(self.datum_1)

        self.assertEqual(buffer.n_trajs, 3)
        self.assertEqual(len(buffer), 10)

    def testLimitless(self):
        MAX_SIZE = -1
        shapes = [(2,), ()]

        buffer = TrajBuffer(MAX_SIZE, shapes) 
        for i in range(1000):
            buffer.add_datum(self.datum_1)

        buffer.add_datum(self.datum_2, True)
        for i in range(1000):
            buffer.add_datum(self.datum_2)

        self.assertEqual(buffer.n_trajs, 2)
        self.assertEqual(len(buffer), 2001)

    def testFillCopy(self):
        MAX_SIZE = 32
        BATCH_SIZE = 200
        shapes = [(2,), ()]
        buffer = TrajBuffer(MAX_SIZE, shapes)
        t11 = np.random.randn(MAX_SIZE, *shapes[0])
        t12 = np.random.randn(MAX_SIZE, *shapes[1])
        traj1 = [t11, t12]
        traj2 = [t11.copy(), t12.copy()]

        buffer.add_traj(traj1)
        self.assertEqual(len(buffer), MAX_SIZE)
        buffer.add_traj(traj2)
        self.assertEqual(len(buffer), MAX_SIZE)
        batch = buffer.get_batch(BATCH_SIZE) # len(shapes) x N x shapes[i]
        self.assertEqual(len(batch), len(shapes))
        self.assertEqual(batch[0].shape[0], BATCH_SIZE)

    def testFillMixed(self):
        MAX_SIZE = 32
        shapes = [(2,), ()]
        buffer = TrajBuffer(MAX_SIZE, shapes)
        t11 = np.random.randn(MAX_SIZE // 2, *shapes[0])
        t12 = np.random.randn(MAX_SIZE // 2, *shapes[1])
        traj1 = [t11, t12]
        traj2 = [t11.copy(), t12.copy()]

        buffer.add_traj(traj1)
        self.assertEqual(len(buffer), MAX_SIZE // 2)

        buffer.add_datum(self.datum_1)
        self.assertEqual(len(buffer), MAX_SIZE // 2 + 1)
        buffer.add_traj(traj2)
        self.assertEqual(len(buffer), MAX_SIZE)
        self.assertEqual(buffer.n_trajs, 3)

    def testGetTrajBatchSimple(self):
        MAX_SIZE = 10
        BATCH_SIZE = 4
        TRAJ_LEN = 3
        shapes = [(2,), ()]
        buffer = TrajBuffer(MAX_SIZE, shapes) 
        buffer.add_datum(self.datum_1)
        buffer.add_datum(self.datum_2)
        buffer.add_datum(self.datum_1)
        buffer.add_datum(self.datum_2)
        batch = buffer.get_traj_batch(BATCH_SIZE, TRAJ_LEN) # len(shapes) x N x T x shapes[i]

        self.assertEqual(len(batch), len(shapes))
        
        for traj_part, shape in zip(batch, shapes):
            self.assertEqual(traj_part.shape[2:], shape)
            self.assertEqual(traj_part.shape[1], TRAJ_LEN)
            self.assertEqual(traj_part.shape[0], BATCH_SIZE)

    def testGetTrajBatchCorrect(self):
        MAX_SIZE = 32
        BATCH_SIZE = 500 # high enough whp
        TRAJ_LEN = 3
        shapes = [(2,), ()]
        buffer = TrajBuffer(MAX_SIZE, shapes) 
        buffer.add_datum(self.datum_1)
        buffer.add_datum(self.datum_2)
        buffer.add_datum(self.datum_1)
        buffer.add_datum(self.datum_2)

        buffer.add_datum(self.datum_1, True)
        buffer.add_datum(self.datum_1)
        buffer.add_datum(self.datum_1)
        buffer.add_datum(self.datum_1)

        buffer.add_datum(self.datum_2, True)
        buffer.add_datum(self.datum_2)
        buffer.add_datum(self.datum_2)
        buffer.add_datum(self.datum_2)

        buffer.add_traj(self.traj_1) # an invalid trajectory to sample
        buffer.add_traj(self.traj_2)

        batch = buffer.get_traj_batch(BATCH_SIZE, TRAJ_LEN) # len(shapes) x N x T x shapes[i]

        unique_firsts = np.unique(batch[0], axis=0)
        self.assertEqual(unique_firsts.shape[0], 5)

    def testGetBatchSimple(self):
        MAX_SIZE = 10
        BATCH_SIZE = 4
        shapes = [(2,), ()]
        buffer = TrajBuffer(MAX_SIZE, shapes) 
        buffer.add_datum(self.datum_1)
        buffer.add_datum(self.datum_2)
        buffer.add_datum(self.datum_1)
        buffer.add_datum(self.datum_2)
        batch = buffer.get_batch(BATCH_SIZE) # len(shapes) x N x shapes[i]

        self.assertEqual(len(batch), len(shapes))
        self.assertEqual(batch[0].shape[0], BATCH_SIZE)

    def testGetBatchCorrect(self):
        MAX_SIZE = 32
        BATCH_SIZE = 500
        shapes = [(2,), ()]
        buffer = TrajBuffer(MAX_SIZE, shapes)
        traj = [np.random.randn(MAX_SIZE, *shapes[0]), np.random.randn(MAX_SIZE, *shapes[1])]
        buffer.add_traj(traj)
        batch = buffer.get_batch(BATCH_SIZE) # len(shapes) x N x shapes[i]

        unique_firsts = np.unique(batch[0], axis=0)
        self.assertEqual(unique_firsts.shape[0], MAX_SIZE)
    
    def testGetBatchMultiple(self):
        MAX_SIZE = 64
        BATCH_SIZE = 200
        shapes = [(2,), ()]
        buffer = TrajBuffer(MAX_SIZE, shapes)
        traj = [np.random.randn(MAX_SIZE // 2, *shapes[0]), np.random.randn(MAX_SIZE // 2, *shapes[1])]

        buffer.add_traj(traj)
        buffer.add_traj(deepcopy(traj))
        batch = buffer.get_batch(BATCH_SIZE) # len(shapes) x N x shapes[i]

        self.assertEqual(len(batch), len(shapes))
        self.assertEqual(batch[0].shape[0], BATCH_SIZE)


    def testGetAll(self):
        buffer = TrajBuffer(10, [(2,), ()]) 
        buffer.add_datum(self.datum_1)
        buffer.add_datum(self.datum_2)

        all_data = buffer.get_all()

        self.assertTrue((all_data[0][0] == self.datum_1[0]).all())
        self.assertTrue((all_data[1][0] == self.datum_1[1]).all())
        self.assertTrue((all_data[0][1] == self.datum_2[0]).all())
        self.assertTrue((all_data[1][1] == self.datum_2[1]).all())

        buffer.add_traj(self.traj_2)

        all_data = buffer.get_all()
        self.assertTrue((all_data[0][-len(self.traj_2[0]):] == self.traj_2[0]).all())
        print(all_data)
        self.assertTrue((all_data[1][-len(self.traj_2[1]):] == self.traj_2[1]).all())




if __name__ == '__main__':
    unittest.main()
