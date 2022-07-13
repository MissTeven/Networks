import numpy as np

if __name__ == '__main__':
    stride = 1
    core_size = 3
    input_shape = (3, 5, 5)
    input_data = np.random.randint(0, 255, input_shape)
    output_channel = 2
    print(input_data)
    slices = []
    input_channel, input_h, input_w = input_data.shape
    for r in range(int((input_h - core_size) / stride + 1)):
        for c in range(int((input_w - core_size) / stride + 1)):
            slices.append(input_data[0:3, r:r + core_size, c:c + core_size])

    print(np.asarray(slices).shape, slices)
    print("===============================================================")
    t = slices
    slices = []
    for i in range(output_channel):
        slices.append(t)
    slices = np.asarray(slices)
    print(slices.shape, slices)
    print("===============================================================")
    core = np.random.ranf((output_channel, 1, input_channel, core_size, core_size))
    print(core.shape, core)
    result = core * slices
    print(result.shape, result)
    print("===============================================================")
    result = np.asarray(result).reshape(
        (output_channel, int((input_h - core_size) / stride + 1), int((input_w - core_size) / stride + 1),
         input_channel, core_size, core_size))
    print(result.shape, result)
    print("===============================================================")
    result = np.sum(result, axis=(3, 4, 5))
    print(result.shape, result)
