import scipy
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np


# 读取存储为txt文件的数据
def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.readline()
        data = raw_data.split(" ")  # [-1:1]是为了去除文件中的前后中括号"[]"

    return np.asfarray(data[:-1])


if __name__ == "__main__":
    train_loss_path = "result/loss_bs4_lr1e-3.txt"  # 存储文件路径

    y_train_loss = data_read(train_loss_path)  # loss值，即y轴
    x_train_loss = range(len(y_train_loss))  # loss的数量，即x轴

    plt.figure()

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('iters')  # x轴标签
    plt.ylabel('loss')  # y轴标签

    # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
    # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
    plt.scatter(x_train_loss, y_train_loss,s=2)
    y_smooth = scipy.signal.savgol_filter(y_train_loss, 53, 3)
    plt.plot(x_train_loss,y_smooth,color='red')
    # plt.legend()
    plt.title('Loss curve')
    plt.show()
