from abc import ABC

import numpy as np
from base import *


class Activator(CalcOperator, ABC):
    def onUpdate(self, lr=0.01, batch=1):
        pass


class SigmoidActivator(Activator):
    def __init__(self):
        super().__init__()
        self.inputs = None
        self.outputs = None

    def onForward(self, input_data):
        print(self, "onForward》》》》》")
        print("input:")
        print(input_data)
        self.inputs = input_data
        self.outputs = 1.0 / (1.0 + np.exp(-1 * self.inputs))
        print("output:")
        print(self.outputs)
        print(self, "《《《《《onForward")
        return self.outputs

    def onBackward(self, loss):
        return np.multiply(loss, np.multiply(self.outputs, 1 - self.outputs))

    def onUpdate(self, lr=0.01, batch=1):
        pass


class TanhActivator(Activator):
    def __init__(self):
        super().__init__()
        self.inputs = None
        self.outputs = None

    def onForward(self, input_data):
        print(self, "onForward》》》》》")
        print("input:")
        print(input_data)
        self.inputs = input_data
        self.outputs = 2.0 / (1.0 + np.exp(-2 * self.inputs)) - 1.0
        print("output:")
        print(self.outputs)
        print(self, "《《《《《onForward")
        return self.outputs

    def onBackward(self, loss):
        return np.multiply(loss, 1 - np.multiply(self.outputs, self.outputs))

    def onUpdate(self, lr=0.01, batch=1):
        pass


class ReluActivator(Activator):
    def __init__(self):
        super().__init__()
        self.inputs = None
        self.outputs = None

    def onForward(self, input_data):
        print(self, "onForward》》》》》")
        print("input:")
        print(input_data)
        self.inputs = np.asmatrix(input_data)
        self.outputs = np.maximum(0, self.inputs)
        print("output:")
        print(self.outputs)
        print(self, "《《《《《onForward")
        return self.outputs

    def onBackward(self, loss):
        return np.matmul(loss, (self.outputs > 0) * 1.0)

    def onUpdate(self, lr=0.01, batch=1):
        pass
