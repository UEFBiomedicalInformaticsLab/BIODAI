import numpy as np

def proximity_matrix(rf_predictor, data_x, normalize:bool = True):
    terminals = rf_predictor.apply(data_x)
    n_trees = terminals.shape[1]

    res = np.zeros((data_x.shape[0], data_x.shape[0]))

    for i in range(n_trees):
        a = terminals[:, i]
        res += np.equal.outer(a, a)

    if normalize:
        res /= n_trees

    return res