from pool import *
import numpy as np


class MaxPool(Pool):
    def onPool(self, input_col):
        return np.max(input_col, axis=1)
