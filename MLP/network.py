import abc

from learning_rate import ConstantLearningRate
from loss import QuadraticLoss


class Network(abc.ABC):
    def __init__(self, loss=QuadraticLoss(), lr=ConstantLearningRate()):
        self.loss = loss
        self.lr = lr

    @abc.abstractmethod
    def train(self, samples, label, batch_size=1, cycle=100):
        pass

    @abc.abstractmethod
    def test(self, samples, label):
        pass

    @abc.abstractmethod
    def predict(self, sample):
        pass
