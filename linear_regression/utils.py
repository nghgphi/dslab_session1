import numpy as np

def normalize_and_add_ones(x):
    x_max = np.array([[np.amax(x[:, column])
                     for column in range(x.shape[1])]
                     for _ in range(x.shape[0])])
    x_min = np.array([[np.amin(x[:, column])
                     for column in range(x.shape[1])]
                     for _ in range(x.shape[0])])
    
    x_normalized = (x - x_min) / (x_max - x_min)

    ones = np.array([[1] for _ in range(x.shape[0])])

    return np.column_stack((ones, x_normalized))



