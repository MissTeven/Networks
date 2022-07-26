from pool import *
import numpy as np


class MinPool(Pool):
    def onPool(self, input_col):
        return np.min(input_col, axis=1)
