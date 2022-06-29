from base.calcoperator import CalcOperator
import numpy as np


class Normalize(CalcOperator):
    def __init__(self):
        super().__init__()
        self.mean = 0
        self.std = 0

    def onForward(self, input_data):
        print(self, "onForward》》》》》")
        print("input:")
        print(input_data)
        input_data = np.asmatrix(input_data)
        self.mean = input_data.mean()
        self.std = input_data.std()
        outputs = (input_data - self.mean) / self.std
        print("output:")
        print(outputs)
        print(self, "《《《《《onForward")
        return outputs

    def onBackward(self, loss):
        return loss

    def onUpdate(self, lr=0.01):
        pass
