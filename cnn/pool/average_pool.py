from pool import *
import numpy as np
from cnn import utils


class AveragePool(Pool):
    def onPool(self, input_col):
        return np.mean(input_col, axis=1)

    def onBackward(self, loss):
        # 对于输入矩阵中未被选中的元素损失为0，对于被选中的元素损失对应着loss相应的元素值
        loss = np.asarray(loss).transpose(0, 2, 3, 1)  # (N,out_h,out_w,C)
        N, out_h, out_w, C = loss.shape
        loss = loss.reshape((N * out_h * out_w * C, -1))  # (N*out_h*out_w,C)
        loss = loss / (self.filter_shape[0] * self.filter_shape[1])
        # self.pool_before_col的shape为
        col = np.ones(shape=self.pool_before_col.shape).reshape(
            (N * out_h * out_w * C, -1))  # (N*out_h*out_w,C*filter_h*filter_w)
        col = col * loss
        col = col.reshape((N * out_h * out_w, -1))
        col = utils.col2img(col, self.input.shape, filter_h=self.filter_shape[0], filter_w=self.filter_shape[1],
                            stride=self.filter_shape, pad=self.pad)
        return col
