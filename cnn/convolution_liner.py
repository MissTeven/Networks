from base.calcoperator import CalcOperator
import numpy as np


class ConvolutionLiner(CalcOperator):
    def __init__(self, stride, padding, core_size, output_channel):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.core_size = core_size
        self.output_channel = output_channel
        self.weights = None
        self.biases = np.asmatrix(np.random.uniform(-1, 1, (output_channel, 1)))

    def onForward(self, input_data):
        input_data = np.asmatrix(input_data)  # c*h*w
        input_data = np.pad(input_data, self.padding)  # c*(h+2*p)*(w+2*p)
        input_channel, input_h, input_w = input_data.shape
        if self.weights is None:
            self.weights = np.asmatrix(
                np.random.uniform(-1, 1, (self.output_channel, input_channel, self.core_size, self.core_size)))
        slices = []
        for i in range(input_channel):
            for r in range((input_h - self.core_size) / self.stride + 1):
                for c in range((input_w - self.core_size) / self.stride + 1):
                    slices.append(input_data[i, r:r + self.core_size, c:c + self.core_size])
        self.weights * slices

        output_data = self.weights * input_data + self.biases
        pass

    def onBackward(self, loss):
        pass

    def onUpdate(self, lr=0.01):
        pass
