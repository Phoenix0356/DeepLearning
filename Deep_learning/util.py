import matplotlib.pyplot as plt
import numpy as np
def draw_plot(test_data,predictions,dot_num,losses):
    len_data=len(test_data)//dot_num*dot_num
    dot_distant = len_data // dot_num
    x = list(range(0,dot_num))
    y_true = [test_data[i][-1] for i in range(0, len_data, dot_distant)]
    y_prediction = [predictions[i] for i in range(0, len_data, dot_distant)]


    fig, axs = plt.subplots(1, 2)
    axs[0].set_title('fit result plot')
    axs[1].set_title('gradient descent plot ')

    # 在第一个子图中绘制散点图和折线图
    axs[0].scatter(x, y_true, color='red')
    axs[0].plot(x, y_true, color='red')

    axs[0].scatter(x, y_prediction, color='blue')
    axs[0].plot(x, y_prediction, color='blue')

    plot_x = np.arange(len(losses))
    plot_y = np.array(losses)
    axs[1].plot(plot_x, plot_y)
    plt.show()