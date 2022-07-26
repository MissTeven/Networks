from pool import *
import numpy as np


class AveragePool(Pool):
    def onPool(self, input_col):
        return np.mean(input_col, axis=1)
