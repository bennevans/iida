import numpy as np

def get_ranges(low, high, n_stripes):
    if n_stripes <= 1:
        return np.array([[low, high]]), np.array([[low, high]])

    widths = (high - low) / n_stripes
    ranges = []
    for stripe in range(n_stripes):
        ranges.append((low + stripe * widths, low + (stripe+1) * widths))
    return np.array(ranges)

def sample_ranges(ranges, n=1):
    n_out_ranges = len(ranges)
    picked_stripes = np.random.randint(n_out_ranges, size=n)
    picked_ranges = ranges[picked_stripes]
    fovs = np.random.uniform(low=picked_ranges[:, 0], high=picked_ranges[:, 1], size=n)
    return fovs

def sample_fovs(fov_config):
    fov_values = {}
    for k, v in fov_config.items():
        fov_values[k] = np.random.uniform(v['low'], v['high'])
    return fov_values

def any_overlap(a, b, ret_same=False):
    """
    checks if there's any overlap between two np arrays
    a and b should be (N, d) and (N', d) respectively
    """
    set_a = set([tuple(x) for x in a])
    set_b = set([tuple(x) for x in b])
    set_intersect = [x for x in set_a & set_b]
    overlap =  len(set_intersect) > 0

    if ret_same:
        return overlap, set_intersect
    return overlap