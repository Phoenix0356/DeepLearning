import matplotlib.pyplot as plt
import numpy as np


x_axis_data = ['softmax','relu','tanh','sigmoid','linear'] # x轴数据
y_axis_data = [0.9545999765396118,
               0.09799999743700027,
               0.09799999743700027,
               0.9537000060081482,
               0.0982000008225441]
plt.plot(x_axis_data, y_axis_data, 'b*--', alpha=0.5, linewidth=1, label='acc') # 'b*--'表示蓝色虚线，数据点为星形标注
plt.legend() # 显示图例
plt.xlabel('hyperParameter_value') # x轴标签
plt.ylabel('accuracy/%') # y轴标签
plt.show() # 显示图像
