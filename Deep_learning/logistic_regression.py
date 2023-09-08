import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import util



# def gender_converter(gender):
#     if gender == 'Male':
#         return 1
#     elif gender == 'Female':
#         return 0


def load_data():
    #Social_Network_Ads.csv
    df = pd.read_csv('D:/MyDataSet/logistic_regression/framingham.csv',
                     #converters={'Gender': gender_converter},
                     dtype=np.float32)
    #df = df.iloc[:, 1:]
    df = df.dropna()
    data = df.to_numpy()

    train_ratio = 0.8
    offset = int(train_ratio * data.shape[0])
    train_set = data[:offset]

    maximums, minimums = train_set.max(axis=0), train_set.min(axis=0)

    for i in range(train_set.shape[1]):
        data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])

    train_set = data[:offset]
    test_set = data[offset:]
    print(train_set)
    return train_set, test_set


class Network(object):
    def __init__(self, num_of_weights):
        np.random.seed(42)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0

    def activating_function(self, z):
        # sigmoid函数
        return 1 / (1 + np.exp(-z))

    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return self.activating_function(z)

    def loss(self, z, y):
        return -1.0*(1.0/y.size)*np.sum(y * np.log(z) + (1 - y) * np.log(1 - z))

    def gradient(self, x, y):
        z = self.forward(x)
        gradient_w = (z - y) * x
        gradient_b = z - y

        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]

        gradient_b = np.mean(gradient_b)
        return gradient_w, gradient_b

    def update(self, gradient_w, gradient_b, eta):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b

    def train(self, x, y, num_of_iteration, eta):
        losses = []
        for i in range(num_of_iteration):
            z = self.forward(x)
            L = self.loss(z, y)
            gradient_w, gradient_b = self.gradient(x, y)
            self.update(gradient_w, gradient_b, eta)
            losses.append(L)
            if (i + 1) % 10 == 0:
                print('iter {}, loss{}'.format(i, L))
        return losses


if __name__ == '__main__':
    train_set, test_set = load_data()
    x=train_set[:,:-1]
    y=train_set[:,-1:]

    net=Network(x.shape[1])
    num_iterations=10000

    losses=net.train(x,y,num_iterations,0.08)

    predictions = net.forward(test_set[:,:-1])
    threshold = 0.5
    predictions_binary = np.where(predictions >= threshold, 1, 0)
    #print(predictions_binary)

    accuracy = np.mean(predictions_binary == test_set[:, -1])
    print('准确率: {:.2f}%'.format(accuracy * 100))

    util.draw_plot(test_set,predictions_binary,8,losses)




