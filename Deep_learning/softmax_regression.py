#import os
#print(os.getcwd())
import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
def load_mnist(path,data_scale, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
    # part_images=images[:data_scale]
    # part_lables=labels[:data_scale]

    return images[:data_scale], labels[:data_scale]


def load_dataset(file_path):
    dataMat = []
    labelMat = []
    fr = open(file_path)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def train(data_arr, label_arr, n_class, iters=1000, alpha=0.1, lam=0.01):
    '''
    @description: softmax 训练函数
    @param {type}
    @return: theta 参数
    '''
    n_samples, n_features = data_arr.shape
    n_classes = n_class
    # 随机初始化权重矩阵
    weights = np.random.rand(n_class, n_features)
    # 定义损失结果
    all_loss = list()
    # 计算 one-hot 矩阵
    y_one_hot = one_hot(label_arr, n_samples, n_classes)
    for i in range(iters):
        # 计算 m * k 的分数矩阵
        scores = np.dot(data_arr, weights.T)
        # 计算 softmax 的值
        probs = softmax(scores)
        # 计算损失函数值
        loss = - (1.0 / n_samples) * np.sum(y_one_hot * np.log(probs))
        all_loss.append(loss)
        # 求解梯度
        dw = -(1.0 / n_samples) * np.dot((y_one_hot - probs).T, data_arr) + lam * weights
        dw[:, 0] = dw[:, 0] - lam * weights[:, 0]
        # 更新权重矩阵
        weights = weights - alpha * dw
    return weights, all_loss


def softmax(scores):
    # 计算总和
    sum_exp = np.sum(np.exp(scores), axis=1, keepdims=True)
    softmax = np.exp(scores) / sum_exp
    return softmax


def one_hot(label_arr, n_samples, n_classes):
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), label_arr.T] = 1
    return one_hot


def predict(test_dataset, label_arr, weights):
    scores = np.dot(test_dataset, weights.T)
    probs = softmax(scores)
    return np.argmax(probs, axis=1).reshape((-1, 1))


if __name__ == "__main__":
    scale=100
    data_arr, label_arr = load_mnist('../fashion-mnist', scale,kind='train')
    test_data_arr, test_label_arr = load_mnist('../fashion-mnist',scale, kind='t10k')


    # gen_dataset()
    #data_arr, label_arr = load_dataset('train_dataset.txt')
    data_arr = np.array(data_arr)
    label_arr = np.array(label_arr).reshape((-1, 1))
    weights, all_loss = train(data_arr, label_arr, n_class=10)

    # 计算预测的准确率
    #test_data_arr, test_label_arr = load_dataset('test_dataset.txt')
    test_data_arr = np.array(test_data_arr)
    test_label_arr = np.array(test_label_arr).reshape((-1, 1))
    n_test_samples = test_data_arr.shape[0]
    y_predict = predict(test_data_arr, test_label_arr, weights)
    accuray = np.sum(y_predict == test_label_arr) / n_test_samples
    print("准确率: ",accuray)

    # 绘制损失函数
    fig = plt.figure(figsize=(8, 5))
    plt.xlim((-10, 10))
    plt.ylim((-10, 10))
    plt.plot(np.arange(1000), all_loss)
    # for i in range(len(all_loss)):
    #     print([all_loss[i]])
    plt.title("Development of loss during training")
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.show()




