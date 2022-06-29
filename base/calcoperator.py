import abc


# 接口类
class CalcOperator(abc.ABC):
    def __init__(self):
        self.next = None
        self.previous = None

    @abc.abstractmethod
    def onForward(self, input_data):
        pass

    def forward(self, input_data):
        if self.next is None:
            return self.onForward(input_data)
        else:
            return self.next.forward(self.onForward(input_data))

    @abc.abstractmethod
    def onBackward(self, loss):
        pass

    def backward(self, loss):
        if self.previous is None:
            return self.onBackward(loss)
        else:
            return self.previous.backward(self.onBackward(loss))

    @abc.abstractmethod
    def onUpdate(self, lr=0.01):
        pass

    def update(self, lr=0.01):
        self.onUpdate(lr)
        if self.next is not None:
            self.next.update(lr)

    def setNext(self, next_coperator):
        self.next = next_coperator
        next_coperator.previous = self
        return self.next
