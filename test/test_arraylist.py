import unittest
import numpy as np

from varyingsim.util.arraylist import ArrayList

# TODO: a[:, 1].shape fails

class ArrayListTest(unittest.TestCase):

    def testInit(self):
        shape = (6,7,8)
        l = ArrayList(shape)

    def testAppendNumpySimple(self):
        datum1 = np.array([[1,2,3],[4,5,6.]])
        datum2 = np.array([[0,0,0],[0,0,0.]])

        shape = (2,3)

        l = ArrayList(shape)
        l.append(datum1)
        l.append(datum2)

        self.assertEqual(2, len(l))
        
        oneEq = (datum1 == l[0]).all()
        twoEq = (datum2 == l[1]).all()
        oneNeq = (datum1 != l[1]).all()
        twoNeq = (datum2 != l[0]).all()

        self.assertTrue(oneEq)
        self.assertTrue(twoEq)
        self.assertTrue(oneNeq)
        self.assertTrue(twoNeq)

    def testAppendExpand(self):
        shape = (10,6,7)
        l = ArrayList(shape)
        SIZE = 200

        for i in range(SIZE):
            l.append(np.random.randn(*shape))
        
        self.assertEqual(SIZE, len(l))
        self.assertEqual(l.buffer.shape[0], 256)
    
    def testAppendMany(self):
        shape = (2,3)
        N = 100
        l = ArrayList(shape)
        data = np.random.randn(N, *shape)
        l.appendMany(data)

        self.assertEqual(N, len(l))
        self.assertTrue((l == data).all())

    def testDifferentMultiplier(self):
        shape = (1000,)
        l = ArrayList(shape, init_size=2, multiplier=1.1)

        l.append(np.zeros(shape))
        l.append(np.zeros(shape))
        l.append(np.zeros(shape))

        self.assertEqual(len(l), 3)

        l.append(np.zeros(shape))
        l.append(np.zeros(shape))
        l.append(np.zeros(shape))

        self.assertEqual(len(l), 6)

    def testAccess(self):
        shape = (2, 3)
        datum1 = np.random.randn(*shape)
        datum2 = np.random.randn(*shape)
        datum3 = np.random.randn(*shape)

        l = ArrayList(shape)
        l.append(datum1)

        self.assertTrue((datum1 == l).all())
        self.assertFalse((datum2 == l).all())

        l.append(datum2)
        l.append(datum3)

        self.assertTrue((l[0] == datum1).all())
        self.assertTrue((l[1] == datum2).all())
        self.assertTrue((l[2] == datum3).all())

    def testPop(self):
        shape = (2, 3)
        datum1 = np.random.randn(*shape)
        datum2 = np.random.randn(*shape)
        datum3 = np.random.randn(*shape)
        datum4 = np.random.randn(*shape)

        l = ArrayList(shape, init_size=2)
        l.append(datum1)
        l.append(datum2)
        
        self.assertEqual(2, len(l))
        
        ret = l.pop()
        
        self.assertTrue((datum1 == ret).all())
        self.assertEqual(1, len(l))
        self.assertTrue((datum2 == l).all())

        l.append(datum3)
        l.append(datum4)
        self.assertEqual(3, len(l))
        self.assertTrue((l[0] == datum2).all())
        self.assertTrue((l[1] == datum3).all())
        self.assertTrue((l[-1] == datum4).all())
        
        ret = l.pop()
        self.assertTrue((datum2 == ret).all())
        ret = l.pop()
        self.assertTrue((datum3 == ret).all())
        ret = l.pop()
        self.assertTrue((datum4 == ret).all())

        self.assertEqual(0, len(l))

        exc = False
        try:
            l.pop()
        except:
            exc = True
        
        self.assertTrue(exc)

        



if __name__ == '__main__':
    unittest.main()
