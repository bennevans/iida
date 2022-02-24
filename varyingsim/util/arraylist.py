

import numpy as np

class ArrayList:
    def __init__(self, shape, init_size=128, multiplier=2.0):

        if multiplier < 1:
            raise Exception("multiplier must be greater than 1! got {}".format(multiplier))

        self.shape = shape
        self.init_size = init_size
        self.multiplier = multiplier

        self.buffer = np.zeros((init_size, *shape))
        self.start_idx = 0
        self.end_idx = 0
        # self.cur_size = 0

    @property
    def cur_size(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, key):
        return self.buffer[self.start_idx:self.end_idx][key]

    def __setitem__(self, key, value):
        self.buffer[self.start_idx:self.end_idx][key] = value

    def expandBuffer(self, N):
        if self.end_idx + N >= len(self.buffer):
            new_size = int(self.buffer.shape[0]*self.multiplier)
            new_size = max(new_size, self.buffer.shape[0] + N)
            new_buffer = np.zeros((new_size, *self.shape))
            new_buffer[:self.cur_size] = self.buffer[self.start_idx:self.end_idx]
            self.buffer = new_buffer
            self.end_idx = self.end_idx - self.start_idx
            self.start_idx = 0


    def append(self, value):
        if value.shape != self.shape:
            raise Exception("shapes don't match! expected {} got {}".format(self.shape, value.shape))
        
        self.expandBuffer(1)
        # if self.end_idx == len(self.buffer):
        #     new_size = int(self.buffer.shape[0]*self.multiplier)
        #     new_size = max(new_size, self.buffer.shape[0] + 1)
        #     new_buffer = np.zeros((new_size, *self.shape))
        #     new_buffer[:self.buffer.shape[0]] = self.buffer
        #     self.buffer = new_buffer

        self.buffer[self.end_idx] = value
        self.end_idx += 1
    
    def appendMany(self, value):
        if value.shape[1:] != self.shape:
            raise Exception("shapes don't match! expected {} got {}".format(self.shape, value.shape))

        N = value.shape[0]
        self.expandBuffer(N)

        # if self.end_idx + N >= len(self.buffer):
        #     new_size = int(self.buffer.shape[0]*self.multiplier)
        #     new_size = max(new_size, self.buffer.shape[0] + 1)
        #     new_buffer = np.zeros((new_size, *self.shape))
        #     new_buffer[:self.buffer.shape[0]] = self.buffer
        #     self.buffer = new_buffer

        self.buffer[self.end_idx:self.end_idx+N] = value
        self.end_idx += N
        
    def pop(self):
        if self.start_idx == self.end_idx:
            raise Exception("cannot pop() with {} elements".format(len(self)))
        ret = self.buffer[self.start_idx]
        self.start_idx += 1
        return ret

    def __repr__(self):
        return self.buffer[self.start_idx:self.end_idx].__repr__()

    def __len__(self):
        return self.cur_size

    def __eq__(self, other):
        return np.equal(self.buffer[self.start_idx:self.end_idx], np.expand_dims(other, 0))


if __name__ == '__main__':
    shape = (5, 6)
    a = ArrayList(shape)
    data = np.random.randn(5, 6)
    a.append(data)
    print(a)
