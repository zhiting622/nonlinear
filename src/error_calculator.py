"""
not sure

"""
import numpy as np

def calculate_error(H, x, y):

    err = np.linalg.norm((H @ x - y), 1)
    return err