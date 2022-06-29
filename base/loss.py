import abc
import numpy as np


class Loss(abc.ABC):
    @abc.abstractmethod
    def calcLoss(self, predict, actual):
        pass


class QuadraticLoss(Loss):

    def calcLoss(self, predict, actual):
        predict = np.asmatrix(predict)
        actual = np.asmatrix(actual)
        value = predict - actual
        return 0.5 * (value * value.T).sum()

    def calcLossGrad(self, predict, actual):
        return predict - actual
