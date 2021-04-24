import numpy as np
from utils import normalize_and_add_ones
from numpy.core.fromnumeric import size\

def prepare(path):
    # data = []
    # file_dir = r'data.txt'
    with open(path) as f:
        data = f.readlines()

    f.close()

    data = [x.strip().split() for x in data]

    data = [[float(y) for y in x] for x in data]

    data = np.array(data)

    x = data[:, :-1]
    y = data[:, -1]
    return x, y
# print(data)
# print(normalize_and_add_ones(data))


# ones = np.array([[1] for _ in range(data.shape[0])])

# print(ones)
# print(data.shape)





