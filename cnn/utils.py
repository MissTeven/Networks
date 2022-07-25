import numpy as np


def img2col(img, filter_h, filter_w, stride=1, pad=0):
    """
    :param img: 由（ 数据量，通道，高，长 ）的4维数组构成的输入数据
    :param filter_h:滤波器的高
    :param filter_w:滤波器的宽
    :param stride:滤波器移动步长
    :param pad:边部填充
    :return:
    """
    img = np.asarray(img)
    N, C, H, W = img.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    img = np.pad(img, pad)
    col = np.zeros((N, out_h, out_w, C, filter_h, filter_w))
    for h in range(out_h):
        for w in range(out_w):
            col[:, h, w, :, :, :] = img[:, :,
                                    h * stride: (h * stride + filter_h),
                                    w * stride:(w * stride + filter_w)]
    col = col.reshape(N * out_h * out_w, -1)
    return col


def col2img(col, img_shape, filter_h, filter_w, stride=1, pad=0):
    """
    :param col: 矩阵
    :param img_shape: 图片的尺寸，由（ 数据量，通道，高，长 ）的维数参数
    :param filter_h:滤波器的高
    :param filter_w:滤波器的长
    :param stride:步幅
    :param pad:填充
    :return:
    """
    N, C, H, W = img_shape
    out_h = (H + pad * 2 - filter_h) // stride + 1
    out_w = (W + pad * 2 - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w)
    for h in range(out_h):
        for w in range(out_w):
            img[:, :,
            h * stride: (h * stride + filter_h),
            w * stride:(w * stride + filter_w)] = col[:, h, w, :, :, :]
    return img[:, :, pad:pad + H, pad:pad + W]


if __name__ == '__main__':
    img = np.random.randint(0, 255, size=(1, 3, 5, 5))
    print(img.shape, img)
    result = img2col(img, filter_h=3, filter_w=3, stride=2)
    print(result.shape, result)
    back = col2img(result, img.shape, 3, 3, stride=2, pad=0)
    print(back.shape, back)
