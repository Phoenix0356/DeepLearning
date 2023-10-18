import numpy as np

def conv2d(input_data,kernel,stride=1):

    input_height,input_width,input_channels=input_data.shape
    kernel_height, kernel_width, kernel_channels = kernel.shape

    output_height = (input_height - kernel_height) // stride + 1
    output_width = (input_width - kernel_width) // stride + 1

    output_data = np.zeros((output_height, output_width))

    # 执行多通道卷积操作
    for i in range(0, input_height - kernel_height + 1, stride):
        for j in range(0, input_width - kernel_width + 1, stride):
            for k in range(kernel_channels):
                output_data[i // stride, j // stride] += np.sum(
                    input_data[i:i + kernel_height, j:j + kernel_width, k] * kernel[:, :, k])

    return output_data




