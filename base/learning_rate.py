import abc


class LearningRate(abc.ABC):
    @abc.abstractmethod
    def rate(self, cycle, sample_index):
        pass


class ConstantLearningRate(LearningRate):
    def __init__(self, rate=0.1):
        self.constant_rate = rate

    def rate(self, cycle, sample_index):
        return self.constant_rate
