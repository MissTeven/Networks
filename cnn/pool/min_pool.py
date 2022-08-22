from pool import *
import numpy as np
from cnn import utils


class MinPool(Pool):
    def onPool(self, input_col):
        return np.min(input_col, axis=1)

    def onBackward(self, loss):
        # 对于输入矩阵中未被选中的元素损失为0，对于被选中的元素损失对应着loss相应的元素值
        loss = np.asarray(loss).transpose(0, 2, 3, 1)  # (N,out_h,out_w,C)
        N, out_h, out_w, C = loss.shape

        col = self.output
        col = col.transpose(0, 2, 3, 1).reshape((N * out_h * out_w * C, -1))

        col = self.pool_before_col.reshape((N * out_h * out_w * C, -1)) - col
        col = np.where(col == 0, 1, 0)  # (N*C*out_h*out_w,filter_h*filter_w)

        loss = loss.reshape((N * out_h * out_w * C, -1))
        col = col * loss
        col = col.reshape((N * out_h * out_w, -1))
        col = utils.col2img(col, self.input.shape, filter_h=self.filter_shape[0], filter_w=self.filter_shape[1],
                            stride=self.filter_shape, pad=self.pad)
        return col
