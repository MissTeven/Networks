import numpy as np


class FullConnectedLayer():
    def __init__(self, input_size, output_size, activator):
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator

        self.w = np.random.uniform(-.1, .1, (output_size, input_size))
        self.b = np.zeros((output_size, input_size))

        self.w_grad_total = np.zeros((output_size, input_size))
        self.b_grad_total = np.zeros((output_size, output_size))

    def forward(self, input_data):
        self.inout_data = input_data
        self.output_data = self.activator.forward(self.w * input_data + self.b)

    def backward(self, input_delt):
        self.sen = input_delt * self.activator.backward(self.output_data)
        output_delt = np.dot(self.w.T, self.sen)
        self.w_grad = np.dot(self.sen, self.inout_data.T)
        self.b_grad = self.sen
        self.w_grad_total += self.w_grad
        self.b_grad_total += self.b_grad
        return output_delt

    def update(self, lr, MBGD_mode=0):
        # MBGO_mode==0指的是SGD模式
        # MGBO_mode==1指的是mini_batch模式
        if MBGD_mode == 0:
            self.w -= lr * self.w_grad
            self.b -= lr * self.b_grad
        elif MBGD_mode == 1:
            self.w -= lr * self.w_grad_add
            self.b -= lr * self.b_grad_add
            self.w_grad_add = np.zeros((self.output_size, self.input_size))
            self.b_grad_add = np.zeros((self.output_size, self.output_size))
    
