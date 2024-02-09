import numpy as np
import cell as c


def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred) ** 2).mean()


y_true = np.array([1, 0, 0, 2])
y_pred = np.array([0, 0, 0, 0])

print(mse_loss(y_true, y_pred))


class OurNeuralNetwork:
    def __init__(self):
        weights = np.array([0, 1])
        bias = 0

        # The Neuron class here is from the previous section
        self.h1 = c.Neuron(weights, bias)
        self.h2 = c.Neuron(weights, bias)
        self.o1 = c.Neuron(weights, bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        # The inputs for o1 are the outputs from h1 and h2
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1


network = OurNeuralNetwork()
x = np.array([2, 3])
print(network.feedforward(x))
