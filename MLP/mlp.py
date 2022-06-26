from activator import TanhActivator, ReluActivator
from full_connected_neuron import FullConnectedNeuron
from learning_rate import ConstantLearningRate
from loss import QuadraticLoss
from network import Network
import numpy as np


class MLP(Network):
    def __init__(self, layer_params, loss=QuadraticLoss(), lr=ConstantLearningRate()):
        super().__init__(loss, lr)

        self.layer_header = None
        self.layer_tail = None

        self.loss = loss
        for i in range(len(layer_params) - 1):

            neuron = FullConnectedNeuron(layer_params[i], layer_params[i + 1], activator=ReluActivator())

            if self.layer_tail is None:
                self.layer_tail = neuron
            else:
                self.layer_tail = self.layer_tail.setNext(neuron)

            if self.layer_header is None:
                self.layer_header = self.layer_tail

    def train(self, samples, label, batch_size=1, cycle=100):
        losses = []
        for i in range(cycle):
            loss = 0
            m = 0
            for j in range(len(samples)):
                sample = samples[j]
                loss += self.loss.calcLossGrad(self.predict(sample=sample), label[j])
                m += 1
                self.layer_tail.backward(loss / m)
                if j % batch_size == 0:
                    self.layer_header.update(lr=self.lr.rate(i, j))
                    loss = 0
                    m += 1
            self.layer_header.update(lr=self.lr.rate(i, j))
            losses.append(self.test(samples, label))
        print("loss count", len(losses))
        print("min loss:", np.min(losses))
        print("max loss:", np.max(losses))

    def test(self, samples, label):
        loss = 0
        for i in range(len(samples)):
            predict = self.predict(samples[i])
            loss += self.loss.calcLoss(predict, label[i])
        print("loss:", loss)
        return loss

    def predict(self, sample):
        return self.layer_header.forward(sample.T)
