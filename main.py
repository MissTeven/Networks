# Press the green button in the gutter to run the script.
import numpy as np

from base.activator import SigmoidActivator
from mlp import MLP

if __name__ == '__main__':
    params = [[4, 1], [2, 1], [1, 1]]
    activator = SigmoidActivator()
    net = MLP(params)

    sample = np.mat([[0.1, 0.2, 0.3, 0.4], [0.2, 0.4, 0.6, 0.8]])
    label = np.array([[3], [9]])

    sample = (sample - sample.mean()) / sample.std()
    print("samples normalized:")
    print(sample)

    net.train(sample, label, cycle=100)
