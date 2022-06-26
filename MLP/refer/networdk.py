import numpy as np

from refer.activator import SigmoidActivator
from refer.layer import FullConnectedLayer


class Network():
    def __init__(self, params_array, activator):
        # params_array为层的唯独信息超参数组
        # layers为网络的集合
        self.layers = []
        for i in range(len(params_array) - 1):
            self.layers.append(FullConnectedLayer(params_array[i], params_array[i + 1], activator))

    def predict(self, sample):
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output_data
        return output

    def calc_gradient(self, label):
        delta = self.layers[-1].output_data - label
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta

    def train_one_sample(self, sample, label, lr):
        self.predict(sample)
        Loss = self.loss(self.layers[-1].output_data, label)
        self.calc_gradient(label)
        self.update(lr)
        return Loss

    def train_batch_sample(self, sample_set, label_set, lr, batch):
        Loss = 0.0
        for i in range(batch):
            self.predict(sample_set[i])
            Loss += self.loss(self.layers[-1].output_data, label_set[i])
            self.calc_gradient(label_set[i])
        self.update(lr, 1)

    def update(self, lr, MBGD_mode=0):
        for layer in self.layers:
            layer.update(lr, MBGD_mode)

    def loss(self, pred, label):
        return 0.5 * ((pred - label) * (pred - label)).sum()

    def gradient_check(self, sample, label):
        self.predict(sample)
        self.calc_gradient(label)
        incre = 10e-4
        for layer in self.layers:
            for i in range(layer.w.shape[0]):
                for j in range(layer.w.shape[1]):
                    layer.w[i][j] += incre
                    pred = self.predict(sample)
                    err1 = self.loss(pred, label)
                    layer.w[i][j] -= 2 * incre
                    pred = self.predict(sample)
                    err2 = self.loss(pred, label)
                    layer.w[i][j] += incre
                    pred_grad = (err1 - err2) / (2 * incre)
                    calc_grad = layer.w_grad[i][j]
                    print("weights({},{}):expected-actual:{}-{}".format(i, j, pred_grad, calc_grad))


if __name__ == '__main__':
    params = [2, 2]
    activator = SigmoidActivator()
    net = Network(params, activator)

    data = np.array([[0, 2], [0, 1]])
    label = np.array([[0, 3], [0, 1]])

    for i in range(10000):
        print("iteration:{}".format(i))
        loss = net.train_one_sample(data, label, 2)
        print("loss:{}".format(loss))

    print("input:")
    print(data)
    print("predict:")
    print(net.predict(data))
    print(" true:")
    print(label)

    net.gradient_check(data, label)
