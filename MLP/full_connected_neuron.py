from activator import SigmoidActivator
from calcoperator import CalcOperator
from full_connected_liner import FullConnectedLiner
from normalize import Normalize


class FullConnectedNeuron(CalcOperator):
    def __init__(self, input_shape=None, output_shape=None, activator=SigmoidActivator()):
        super().__init__()
        self.line_layer = FullConnectedLiner(input_shape, output_shape)
        self.activator = activator

        self.line_layer.setNext(self.activator)

    def onForward(self, input_data):
        print(self, "onForward》》》》》")
        print("input:")
        print(input_data)
        output = self.line_layer.forward(input_data)
        print("output:")
        print(output)
        print(self, "《《《《《onForward")
        return output

    def onBackward(self, loss):
        return self.line_layer.backward(loss)

    def onUpdate(self, lr=0.01):
        self.line_layer.update(lr)

    def setNext(self, next_coperator):
        next_coperator.previous = self
        self.next = next_coperator
        return next_coperator
