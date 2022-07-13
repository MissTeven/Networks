import numpy as np

from base.calcoperator import CalcOperator


class FullConnectedLiner(CalcOperator):

    def __init__(self, input_shape=None, output_shape=None):
        super().__init__()
        self.inputs = None
        self.outputs = None

        self.weights = np.asmatrix(np.random.uniform(-.1, .1, (output_shape[0], input_shape[0])))
        self.biases = np.asmatrix(np.random.uniform(-.1, .1, (output_shape[0], output_shape[1])))

        self.weights_grad_total = np.asmatrix(np.zeros((output_shape[0], input_shape[0])))
        self.biases_grad_total = np.asmatrix(np.zeros((output_shape[0], output_shape[1])))

        self.backward_times = 0

    def onForward(self, input_data):
        print(self, "onForward》》》》》")
        print("input:")
        print(input_data)
        self.inputs = input_data
        self.outputs = self.weights * self.inputs + self.biases
        print("output:")
        print(self.outputs)
        print(self, "《《《《《onForward")
        return self.outputs

    def onBackward(self, loss):
        weights_grad = np.dot(loss, self.inputs.T)
        biases_grad = loss

        self.weights_grad_total += weights_grad
        self.biases_grad_total += biases_grad

        self.backward_times += 1

        return np.dot(self.weights.T, loss)

    def onUpdate(self, lr=0.01):
        if self.weights_grad_total is None or self.biases_grad_total is None or self.backward_times == 0:
            return
        self.weights -= lr * (self.weights_grad_total / self.backward_times)
        self.biases -= lr * (self.biases_grad_total / self.backward_times)

        self.weights_grad_total = np.asmatrix(np.zeros(self.weights.shape))
        self.biases_grad_total = np.asmatrix(np.zeros(self.biases.shape))

        self.backward_times = 0
