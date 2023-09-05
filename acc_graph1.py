import scipy
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np


# 读取存储为txt文件的数据
def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.readline()
        data = raw_data.split(" ")  # [-1:1]是为了去除文件中的前后中括号"[]"
    return np.asfarray(data[:-3])


if __name__ == "__main__":
    train_acc_path = r"result/acc_bs4_lr1e-5.txt"  # 存储文件路径
    train_none = "result/acc_bs5_lr1e-5.txt"  # 存储文件路径
    train_noperson = r"result/acc_bs2_lr1e-5.txt"  # 存储文件路径



    y_full = data_read(train_acc_path)  # 训练准确率值，即y轴
    y_none = data_read(train_none)  # 训练准确率值，即y轴
    y_noperson = data_read(train_noperson)  # 训练准确率值，即y轴
    x_train_acc = range(1,len(y_full)+1)  # 训练阶段准确率的数量，即x轴

    plt.figure()

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('epochs')  # x轴标签
    plt.ylabel('accuracy')  # y轴标签

# 以x_train_acc为横坐标，y_train_acc为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
# 增加参数color='red',这是红色。
#     plt.axhline(y=max(y_none),color='blue',linewidth=1, linestyle="dotted")
#     plt.axhline(y=max(y_noperson), color='green', linewidth=1, linestyle="dotted")
#     plt.axhline(y=max(y_full), color='red', linewidth=1, linestyle="dotted")

    plt.plot(x_train_acc, y_none,color='blue', linewidth=1, linestyle="solid", label="batchsize=5")
    plt.plot(x_train_acc, y_noperson, color='green',linewidth=1, linestyle="solid", label="batchsize=2")
    plt.plot(x_train_acc, y_full, color='red',linewidth=1, linestyle="solid", label="batchsize=4")
    plt.legend()
    plt.title('Accuracy curve')
    plt.show()
