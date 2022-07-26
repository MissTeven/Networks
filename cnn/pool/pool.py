from base import *
import numpy as np
from cnn import utils


class Pool(CalcOperator):
    def __init__(self, filter_shape, stride=1, pad=0):
        super().__init__()
        self.filter_shape = filter_shape
        self.stride = stride
        self.pad = pad

        self.input = None
        self.output = None

    def onForward(self, input_data):
        input_data = np.asarray(input_data)
        self.input = input_data

        N, C, H, W = input_data.shape
        out_h = (H + 2 * self.pad - self.filter_shape[0]) // self.filter_shape[0] + 1
        out_w = (W + 2 * self.pad - self.filter_shape[1]) // self.filter_shape[1] + 1
        col = utils.img2col(input_data, filter_h=self.filter_shape[0], filter_w=self.filter_shape[1],
                            stride=self.filter_shape, pad=self.pad)  # (N*out_h*out_w,C*filter_h*filter_w)
        col = col.reshape((N, out_h, out_w, C, self.filter_shape[0], self.filter_shape[1])).transpose(
            (0, 3, 1, 2, 4, 5)).reshape((N * C * out_h * out_w, self.filter_shape[0] * self.filter_shape[1]))
        col = self.onPool(col)
        col = col.reshape((N, C, out_h, out_w))
        self.output = col
        return self.output

    @abc.abstractmethod
    def onPool(self, input_col):
        pass

    def onBackward(self, loss):
        pass

    def onUpdate(self, lr=0.01):
        pass
