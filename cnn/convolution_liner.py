import utils
from base import *


class ConvolutionLiner(CalcOperator):
    def __init__(self, filter_shape, output_channel, stride=1, padding=0):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.filter_shape = filter_shape
        self.output_channel = output_channel
        self.img_slice_col = None
        self.input_data = None
        self.weights = None
        self.weights_col = None
        self.biases = None
        self.output_data = None

        self.weights_grad_total = None
        self.biases_grad_total = None

    def onForward(self, input_data):
        input_data = np.asarray(input_data)
        self.input_data = input_data

        N, C, H, W = input_data.shape
        filter_h, filter_w = self.filter_shape

        output_h = (H + 2 * self.padding - filter_h) // self.stride + 1
        output_w = (W + 2 * self.padding - filter_w) // self.stride + 1

        col = utils.img2col(input_data, filter_h, filter_w, self.stride,
                            self.padding)  # (N*out_h*out_w,C*filter*h*filter_w)

        if self.weights is None:
            self.weights = np.random.ranf((self.output_channel, C, filter_h, filter_w))

        col_w = self.weights.reshape(self.output_channel, -1).T  # (C*filter_h*filter_w,output_channel)

        output_data = np.dot(col, col_w)  # ( N*out_h*out_w,out_chanel)
        output_data = output_data.reshape((N, output_h, output_w, self.output_channel)).transpose(
            (0, 3, 1, 2))  # （N,out_channel,out_h,out_w）

        if self.biases is None:
            self.biases = np.random.ranf((self.output_channel, 1, 1))

        output_data += self.biases
        self.img_slice_col = col  # (N*out_h*out_w,C*filter*h*filter_w)
        self.weights_col = col_w

        self.output_data = output_data  # (N,out_chanel,out_h,out_w)
        return self.output_data

    def onBackward(self, loss):
        """
        :param loss: (N,out_chanel,out_h,out_w)
        :return:
        """
        loss_col = np.asarray(loss)
        N, out_chanel, out_h, out_w = loss_col.shape
        loss_col = loss_col.transpose((0, 2, 3, 1)).reshape(N * out_h * out_w, -1)  # (N*out_h*out_w,out_channel)
        biases_grad = np.sum(loss_col, axis=0).reshape(self.biases.shape)

        col_w_grad = np.dot(loss_col.T, self.img_slice_col)  # (out_channel,C*filter_h*filter_w)

        weights_grad = col_w_grad.reshape(
            (out_chanel, self.input_data.shape[1], self.filter_shape[0], self.filter_shape[1]))

        self.weights_grad_total = weights_grad
        self.biases_grad_total = biases_grad

        loss_col = np.dot(loss_col, self.weights_col.T)  # (N*out_h*out_w,C*filter_h*filter_w)
        return utils.col2img(loss_col, img_shape=self.input_data.shape, filter_h=self.filter_shape[0],
                             filter_w=self.filter_shape[1], stride=self.stride, pad=self.padding)

    def onUpdate(self, lr=0.01):
        self.weights -= lr * self.weights_grad_total/self.input_data.shape[0]
        self.biases -= lr * self.biases_grad_total/self.input_data.shape[0]


if __name__ == '__main__':
    img = np.random.randint(0, 255, size=(4, 3, 5, 5))
    print(img.shape, img)
    conv = ConvolutionLiner(filter_shape=(3, 3), output_channel=2)
    result = conv.forward(img)
    print(result.shape, result)
    loss = np.random.ranf((4, 2, 3, 3)) * 255
    back = conv.backward(loss)
    print(back.shape, back)
    print(conv.biases.shape, conv.biases)
    print(conv.weights.shape, conv.weights)
    conv.update()
    print(conv.biases.shape, conv.biases)
    print(conv.weights.shape, conv.weights)
