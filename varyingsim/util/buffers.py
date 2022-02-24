import numpy as np

from varyingsim.util.arraylist import ArrayList

class FixedBuffer:
    def __init__(self, buf_size, buf_shapes):
        self.buf_size = buf_size
        self.cur_buf_size = 0
        self.cur_buf_idx = 0

        self.buffers = []
        for shape in buf_shapes:
            self.buffers.append(np.zeros((buf_size, *shape)))

    def add_to_buffer(self, data):
        for datum, buffer in zip(data, self.buffers):
            buffer[self.cur_buf_idx] = datum

        if self.cur_buf_size < self.buf_size:
            self.cur_buf_size += 1
        self.cur_buf_idx = (self.cur_buf_idx + 1) % self.buf_size

    def get_buffers(self):
        return self.buffers

    def _get_batch_range(self, start, stop, batch_size):
        rand_idxs = np.random.choice(np.arange(start, stop), batch_size)
        ret = []
        for buffer in self.buffers:
            ret.append(buffer[rand_idxs])
        return ret
    
    def get_batch(self, batch_size):
        return self._get_batch_range(0, self.cur_buf_size, batch_size)
    
    def __len__(self):
        return self.cur_buf_size
    
    def reset(self):
        self.cur_buf_size = 0
        self.cur_buf_idx = 0

class ContinualSplitBuffer:
    def __init__(self, buf_size, buf_shapes, train_val_ratio):
        self.train_val_ratio = train_val_ratio
        train_size = int(buf_size * train_val_ratio)
        self.train_buf = FixedBuffer(train_size , buf_shapes)
        self.val_buf = FixedBuffer(buf_size - train_size, buf_shapes)
        
    @property
    def val_idx(self):
        return int(self.cur_buf_size * self.train_val_ratio)
    
    def add_to_buffer(self, data):
        if np.random.rand() < self.train_val_ratio:
            self.train_buf.add_to_buffer(data)
        else:
            self.val_buf.add_to_buffer(data)
    
    def get_train_batch(self, batch_size):
        return self.train_buf.get_batch(batch_size)
    
    def get_val_batch(self, batch_size):
        return self.val_buf.get_batch(batch_size)
    
    def __len__(self):
        return len(self.val_buf) + len(self.train_buf)
    
    def reset(self):
        self.val_buf.reset()
        self.train_buf.reset()

# TODO: numpify the previous list after we create a new traj?
# TODO: insert numpy trajs before the current
class TrajBuffer:
    def __init__(self, max_buf_size, buf_shapes, numpify_freq=100):
        self.max_buf_size = max_buf_size
        self.cur_buf_size = 0
        self.buf_shapes = buf_shapes

        self.trajectories = [] # a list of trajectories
        # a trajectory is a list of either lists or ndarrays all with the same length
        # and whose shapes correspond to buf_shapes
        self.traj_lens = [0]
        self.new_traj = False
        self.numpify_freq = numpify_freq

    # def add_datum_numpy(self, data):
    #     for i, datum in enumerate(data):
    #         self.trajectories[-1][i] = np.vstack([self.trajectories[-1][i], datum]) # this is super innefficient

    # def add_datum_list(self, data):
    #     for datum, buf in zip(data, self.trajectories[-1]):
    #         buf.append(datum)

    def _add_datum(self, data):
        for i, datum in enumerate(data):
            self.trajectories[-1][i].append(datum)

    def drop_data(self):
        if self.max_buf_size < 0:
            return

        while len(self) > self.max_buf_size: # TODO: could technically speed up
            traj = self.trajectories[0]
            for i in range(len(traj)):
                traj[i].pop()

            self.cur_buf_size -= 1
 
            self.traj_lens[0] -= 1
            if self.traj_lens[0] <= 0:
                if len(self.traj_lens) > 1:
                    self.traj_lens.pop(0)
                self.trajectories.pop(0)

    def add_datum(self, data, new_traj=False):
        """
            lazily appends datum to the trajectory. Converts to numpy when 
            data - a list of np arrays with shapes corresponding to self.buf_shapes
            new_traj - If the data belongs to a new trajectory
        """

        if len(data) != len(self.buf_shapes):
            raise Exception('must give same number of data as buf_shapes! expected: {} got: {}'.format(len(self.buf_shapes), len(data)))
        for datum, shape in zip(data, self.buf_shapes):
            if datum.shape != shape:
                raise Exception('data shape must match buf_shapes! expected: {} got: {}'.format(shape, datum.shape))
        
        new_traj = new_traj or self.new_traj
        self.new_traj = False

        if new_traj:
            self.traj_lens.append(0) # TODO: get rid of initializing with [0] and set self.new_traj = True

        if new_traj or len(self.trajectories) == 0:
            arrs = []
            for datum, shape in zip(data, self.buf_shapes):
                l = ArrayList(shape)
                l.append(datum)
                arrs.append(l)
            self.trajectories.append(arrs)
        else:
            self._add_datum(data)

        self.traj_lens[self.traj_idx] += 1

        self.cur_buf_size += 1

        self.drop_data()


    def add_traj(self, trajectory):
        """
            trajectory - a list of np arrays to add. trajectory cannot be shared memory of another trajectory in the buffer
        """
        N = trajectory[0].shape[0]
        for traj_part, shape in zip(trajectory, self.buf_shapes):
            if traj_part.shape[1:] != shape:
                raise Exception('data shape must match buf_shapes! expected: {} got: {}'.format(shape, traj_part.shape))
            if traj_part.shape[0] != N:
                raise Exception('data must have same length for all inputs! got {} and {}'.format(N, traj_part.shape[0]))
        
        for traj in self.trajectories:
            if type(traj) == np.ndarray and traj == trajectory:
                if np.shares_memory(trajectory[0], traj[0]):
                    raise Exception('cannot pass in same trajectory without copying')

        # TODO: insert the trajectory before the current running one?

        if len(self.traj_lens) == 1 and self.traj_lens[0] == 0:
            self.traj_lens[0] += N
        else:
            self.traj_lens.append(N)

        new_traj = []
        for shape, traj_part in zip(self.buf_shapes, trajectory):
            l = ArrayList(shape)
            l.appendMany(traj_part)
            new_traj.append(l)
        self.trajectories.append(new_traj)

        self.cur_buf_size += N
        self.drop_data()
        self.new_traj = True

    @property
    def n_trajs(self):
        return len(self.trajectories)

    @property
    def traj_idx(self):
        return self.n_trajs - 1

    def get_traj_batch(self, batch_size, traj_len):
        """
            returns a batch of size batch_size x len(self.buf_shapes) x T x self.buf_shapes[i]
            TODO: should this be len(self.buf_shapes) x batch_size x T x self.buf_shapes[i]??
        """
        num_val_traj = np.array([max(0, tl - traj_len + 1) for tl in self.traj_lens])
        num_val = num_val_traj.sum()

        if num_val == 0:
            raise LookupError('not enough valid trajectories in the buffer for length: {}'.format(traj_len))

        rand_idx = np.random.choice(num_val, size=batch_size)

        ps = np.zeros(len(num_val_traj), dtype=np.int)
        for i, v in enumerate(num_val_traj): 
            ps[i] = ps[i-1] + v
        ps_rep = ps.repeat(batch_size, axis=0).reshape(len(num_val_traj), batch_size)
        traj_idxs = (ps_rep > rand_idx).argmax(axis=0)
        start_idxs = num_val_traj[traj_idxs] + rand_idx - ps[traj_idxs]

        ret = [[] for _ in self.buf_shapes]

        for traj_idx, start_idx in zip(traj_idxs, start_idxs):
            selected_traj = self.trajectories[traj_idx]
            for i, traj_part in enumerate(selected_traj):
                # # TODO: necessary?
                # if type(traj_part) == list:
                #     traj_part = np.array(traj_part)
                ret[i].append([traj_part[start_idx:start_idx+traj_len]])
        
        for i in range((len(ret))):
            ret[i] = np.concatenate(ret[i])

        return ret
    
    def get_batch(self, batch_size):
        """
            returns a batch of size len(self.buf_shapes) x batch_size x self.buf_shapes[i]
        """

        rand_idx = np.random.choice(self.cur_buf_size, size=batch_size)

        ps = np.zeros(len(self.traj_lens), dtype=np.int)
        for i, v in enumerate(self.traj_lens): 
            ps[i] = ps[i-1] + v 
        ps_rep = ps.repeat(batch_size, axis=0).reshape(len(self.traj_lens), batch_size)
        traj_idxs = (ps_rep > rand_idx).argmax(axis=0)

        start_idxs = np.array(self.traj_lens)[traj_idxs] + rand_idx - ps[traj_idxs]

        ret = [[] for _ in self.buf_shapes]
        for traj_idx, start_idx in zip(traj_idxs, start_idxs):
            selected_traj = self.trajectories[traj_idx]
            for i, traj_part in enumerate(selected_traj):
                # # TODO: necessary?
                # if type(traj_part) == list:
                #     traj_part = np.array(traj_part)
                ret[i].append(traj_part[start_idx])
        
        for i in range((len(ret))):
            ret[i] = np.array(ret[i])

        return ret
    
    def get_hist(self, hist_size):
        """
            returns a trajectory of the most recent data
        """
        traj = self.trajectories[-1]

        ret = [[] for _ in self.buf_shapes]

        for i, traj_part in enumerate(traj):
            T = len(traj_part)
            if hist_size < T:
                ret[i] = traj_part[-hist_size:]
            else: # pad with zeros
                ret[i] = np.concatenate([np.zeros((hist_size-T, *self.buf_shapes[i])), traj_part])
        
        for i in range(len(ret)):
            # ret[i] = np.expand_dims(np.array(ret[i]), axis=0)
            ret[i] = np.array(ret[i])
        return ret

    def get_all(self):
        """
        returns all trajectories in the buffer
        """
        ret = []
        for traj in self.trajectories:
            for i, traj_part in enumerate(traj):
                if len(ret) <= i:
                    ret.append(traj_part)
                else:
                    ret[i] = np.concatenate([ret[i], traj_part])
        return ret

    def set_new_traj(self):
        self.new_traj = True

    def __len__(self):
        return self.cur_buf_size

    def __repr__(self):
        return "TrajBuffer N: {} n_trajs: {}".format(len(self), self.n_trajs)

    def reset(self):
        self.cur_buf_size = 0
        self.cur_buf_idx = 0
        self.trajectories = []
        self.traj_lens = [0]

class KeyedBuffer(TrajBuffer):

    def __init__(self, keys, max_buf_size, buf_shapes, numpify_freq=100):
        super().__init__(max_buf_size, buf_shapes, numpify_freq=numpify_freq)
        """
            expects all things to be in dictionary form. ie {'qpos': [1, 0, 0]}
            keys - a list of strings of the keys of the buffer
        """
        if len(keys) != len(buf_shapes):
            raise Exception("keys and buf_shapes must have same length!")
        
        self.keys = keys
        self.key_map = dict((k, i) for i, k  in enumerate(keys))

    def check_keys(self, input):
        if input.keys() != set(self.keys):
            raise ValueError("dict keys must match keys: {}".format(self.keys))

    def make_dict(self, batch):
        return dict((k, t) for k, t in zip(self.keys, batch))

    def get_batch(self, batch_size):
        batch = super().get_batch(batch_size)
        return self.make_dict(batch)
    
    def get_traj_batch(self, batch_size, traj_len):
        batch = super().get_traj_batch(batch_size, traj_len)
        return self.make_dict(batch)
    
    def get_all(self):
        all = super().get_all()
        return self.make_dict(all)
        
    def get_hist(self, hist_size):
        batch = super().get_hist(hist_size)
        return self.make_dict(batch)

    def add_datum(self, data, new_traj=False):
        datum = [0] * len(self.keys)
        for k, v in data.items():
            datum[self.key_map[k]] = v
        super().add_datum(datum, new_traj=new_traj)

    def add_traj(self, trajectory):
        traj = [0] * len(self.keys)
        for k, v in trajectory.items():
            traj[self.key_map[k]] = v
        super().add_traj(traj)

class DatumBuffer:
    def __init__(self, max_buf_size, buf_shapes):
        self.max_buf_size = max_buf_size
        self.cur_buf_size = 0
        self.buf_shapes = buf_shapes

        self.trajectories = [] # a list of array lists whose shapes correspond to buf_shapes

        for shape in self.buf_shapes:
            self.trajectories.append([])

    def _add_datum(self, data):
        for i, datum in enumerate(data):
            self.trajectories[i].append(datum)

    def drop_data(self):
        if self.max_buf_size < 0:
            return

        while len(self) > self.max_buf_size: # TODO: could technically speed up
            for traj in self.trajectories:
                traj.pop(0)
            self.cur_buf_size -= 1

    def add_datum(self, data, new_traj=False):
        """
            lazily appends datum to the trajectory. Converts to numpy when 
            data - a list of np arrays with shapes corresponding to self.buf_shapes
            new_traj - If the data belongs to a new trajectory
        """

        if len(data) != len(self.buf_shapes):
            raise Exception('must give same number of data as buf_shapes! expected: {} got: {}'.format(len(self.buf_shapes), len(data)))
        for datum, shape in zip(data, self.buf_shapes):
            if datum.shape != shape:
                raise Exception('data shape must match buf_shapes! expected: {} got: {}'.format(shape, datum.shape))

        self._add_datum(data)
        self.cur_buf_size += 1
        self.drop_data()
    
    def get_batch(self, batch_size):
        """
            returns a batch of size len(self.buf_shapes) x batch_size x self.buf_shapes[i]
        """

        rand_idx = np.random.choice(self.cur_buf_size, size=batch_size)
        ret = [[] for _ in self.buf_shapes]
        
        for idx in rand_idx:
            for i, traj_part in enumerate(self.trajectories):
                ret[i].append(traj_part[idx])

        for i in range((len(ret))):
            ret[i] = np.array(ret[i])

        return ret
    
    def get_hist(self, hist_size):
        """
            returns a trajectory of the most recent data
        """

        ret = [[] for _ in self.buf_shapes]

        for i, traj_part in enumerate(self.trajectories):
            T = len(traj_part)
            if hist_size < T:
                ret[i] = traj_part[-hist_size:]
            else: # pad with zeros
                ret[i] = np.concatenate([np.zeros((hist_size-T, *self.buf_shapes[i])), traj_part])
        
        for i in range(len(ret)):
            # ret[i] = np.expand_dims(np.array(ret[i]), axis=0)
            ret[i] = np.array(ret[i])
        return ret

    def get_all(self):
        """
        returns all trajectories in the buffer
        """
        return self.trajectories

    def __len__(self):
        return self.cur_buf_size

    def __repr__(self):
        return "DatumBuffer N: {}".format(len(self))

    def reset(self):
        self.cur_buf_size = 0
        self.cur_buf_idx = 0
        self.trajectories = []

class KeyedDatumBuffer(DatumBuffer):

    def __init__(self, keys, max_buf_size, buf_shapes):
        super().__init__(max_buf_size, buf_shapes)
        """
            expects all things to be in dictionary form. ie {'qpos': [1, 0, 0]}
            keys - a list of strings of the keys of the buffer
        """
        if len(keys) != len(buf_shapes):
            raise Exception("keys and buf_shapes must have same length!")
        
        self.keys = keys
        self.key_map = dict((k, i) for i, k  in enumerate(keys))

    def check_keys(self, input):
        if input.keys() != set(self.keys):
            raise ValueError("dict keys must match keys: {}".format(self.keys))

    def make_dict(self, batch):
        return dict((k, t) for k, t in zip(self.keys, batch))

    def get_batch(self, batch_size):
        batch = super().get_batch(batch_size)
        return self.make_dict(batch)
    
    def get_traj_batch(self, batch_size, traj_len):
        batch = super().get_traj_batch(batch_size, traj_len)
        return self.make_dict(batch)
    
    def get_all(self):
        all = super().get_all()
        return self.make_dict(all)
        
    def get_hist(self, hist_size):
        batch = super().get_hist(hist_size)
        return self.make_dict(batch)

    def add_datum(self, data):
        datum = [0] * len(self.keys)
        for k, v in data.items():
            datum[self.key_map[k]] = v
        super().add_datum(datum)

    def add_traj(self, trajectory):
        traj = [0] * len(self.keys)
        for k, v in trajectory.items(): 
            traj[self.key_map[k]] = v
        super().add_traj(traj)

class ReservoirBuffer(DatumBuffer):

    # https://openreview.net/pdf?id=B1gTShAct7 algo 3

    def __init__(self, max_buf_size, buf_shapes):
        super().__init__(max_buf_size, buf_shapes)
        self.age = 0

    def add_datum(self, data):
        self.age += 1
        if self.max_buf_size >= self.age:
            super().add_datum(data)
        else:
            j = np.random.randint(self.age)
            if j < self.max_buf_size:
                self.drop_index(j)
                super().add_datum(data)

    def drop_index(self, p):
        for traj in self.trajectories:
            traj.pop(p)
        self.cur_buf_size -= 1


class KeyedReservoirBuffer(ReservoirBuffer):

    def __init__(self, keys, max_buf_size, buf_shapes):
        super().__init__(max_buf_size, buf_shapes)
        """
            expects all things to be in dictionary form. ie {'qpos': [1, 0, 0]}
            keys - a list of strings of the keys of the buffer
        """
        if len(keys) != len(buf_shapes):
            raise Exception("keys and buf_shapes must have same length!")
        
        self.keys = keys
        self.key_map = dict((k, i) for i, k  in enumerate(keys))

    def check_keys(self, input):
        if input.keys() != set(self.keys):
            raise ValueError("dict keys must match keys: {}".format(self.keys))

    def make_dict(self, batch):
        return dict((k, t) for k, t in zip(self.keys, batch))

    def get_batch(self, batch_size):
        batch = super().get_batch(batch_size)
        return self.make_dict(batch)
    
    def get_traj_batch(self, batch_size, traj_len):
        batch = super().get_traj_batch(batch_size, traj_len)
        return self.make_dict(batch)
    
    def get_all(self):
        all = super().get_all()
        return self.make_dict(all)
        
    def get_hist(self, hist_size):
        batch = super().get_hist(hist_size)
        return self.make_dict(batch)

    def add_datum(self, data):
        datum = [0] * len(self.keys)
        for k, v in data.items():
            datum[self.key_map[k]] = v
        super().add_datum(datum)

    def add_traj(self, trajectory):
        traj = [0] * len(self.keys)
        for k, v in trajectory.items(): 
            traj[self.key_map[k]] = v
        super().add_traj(traj)


def test_reservoir():
    shapes = [(1,)]

    resevoir_buf = ReservoirBuffer(100, shapes)
    buf = DatumBuffer(100, shapes)

    data = [np.array([i]) for i in np.arange(1000)]
    for datum in data:
        resevoir_buf.add_datum([datum])
        buf.add_datum([datum])
    
    # plt.hist(resevoir_buf.trajectories[0], bins=50)
    # plt.show()

    # plt.hist(buf.trajectories[0], bins=50)
    # plt.show()
    return resevoir_buf, buf


def test_keyed_reservoir():
    shapes = [(1,)]

    resevoir_buf = KeyedReservoirBuffer(['data'], 100, shapes)

    data = [np.array([i]) for i in np.arange(1000)]
    for datum in data:
        resevoir_buf.add_datum({'data': datum})
    
    # plt.hist(resevoir_buf.trajectories[0], bins=50)
    # plt.show()

    # plt.hist(buf.trajectories[0], bins=50)
    # plt.show()
    return resevoir_buf

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # resevoir_buf, buf = test_reservoir()
    resevoir_buf = test_keyed_reservoir()

    
