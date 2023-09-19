# import os
# print(os.getcwd())
import os
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import util




# def load_mnist(path, data_scale, train_ratio, kind='train'):
#     """Load MNIST data from `path`"""
#     labels_path = os.path.join(path,
#                                '%s-labels-idx1-ubyte.gz'
#                                % kind)
#     images_path = os.path.join(path,
#                                '%s-images-idx3-ubyte.gz'
#                                % kind)
#
#     with gzip.open(labels_path, 'rb') as lbpath:
#         labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
#                                offset=8)
#
#     with gzip.open(images_path, 'rb') as imgpath:
#         images = np.frombuffer(imgpath.read(), dtype=np.uint8,
#                                offset=16).reshape(len(labels), 784)
#     train_num = int(data_scale * train_ratio)
#
#     training_data_set = images[:train_num]
#     training_label_set = labels[:train_num]
#     testing_data_set = images[train_num:data_scale]
#     testing_label_set = labels[train_num:data_scale]
#
#     return training_data_set, training_label_set, testing_data_set, testing_label_set

def load_data(path):
    #Social_Network_Ads.csv
    df = pd.read_csv(path,
                     #converters={'Gender': gender_converter},
                     dtype=np.float32
                     )
    #df = df.iloc[:, 1:]
    df = df.dropna()
    data = df.to_numpy()

    train_ratio = 0.8
    offset = int(train_ratio * data.shape[0])
    train_set = data[:offset]

    maximums, minimums = train_set[:,:-1].max(axis=0), train_set[:,:-1].min(axis=0)

    for i in range(train_set.shape[1]-1):
        data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])

    train_set = data[:offset]
    test_set = data[offset:]

    return train_set ,test_set

def convert_to_one_hot(label_set, num_classes):
    return np.eye(num_classes)[label_set]


class Net_work():
    def __init__(self, num_of_weights,num_of_feature):
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights,num_of_feature)
        self.b = 0.

    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return self.activating_function(z)

    def activating_function(self, z):
        # softmax函数
        z -= np.max(z, axis=1, keepdims=True)
        exp_sum = np.sum(np.exp(z), axis=1, keepdims=True)
        return np.exp(z) / exp_sum

    def loss(self, z, y):
        return -1.0 * (1.0 / y.size) * np.sum(y * np.log(z))

    def gradient(self, x, y):
        z = self.forward(x)
        gradient_w = np.dot((z - y).T,x )/y.size
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (z - y)
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


# (train_set, train_label_set,
#  test_set, test_label_set) = load_mnist('../fashion-mnist', 1000, 0.8)

if __name__ == '__main__':
    train_set,test_set=load_data('D:/MyDataSet/softmax_regression/flag.csv')
    train_label_one_hot = convert_to_one_hot(train_set[:,-1:].astype(int).reshape(-1),3)
    net = Net_work(train_set.shape[1],3)
    losses=net.train(train_set, train_label_one_hot, 2000, 0.01)

    predictions = net.forward(test_set)
    predicted_classes = np.argmax(predictions, axis=1)

    util.draw_plot(test_set,predicted_classes,20,losses)
