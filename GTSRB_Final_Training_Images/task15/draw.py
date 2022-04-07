"""
绘制折线图代码， 库名：matplotlib
"""
import matplotlib
import matplotlib.pyplot as plt
# 处理乱码
from matplotlib.pyplot import MultipleLocator
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
# 设置x刻度的间隔倍数
x_major_locator=MultipleLocator(1000)
# x = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
# y = [96.61, 96.73, 97.52, 98.39, 98.36, 98.33, 97.96, 98.21, 98.04, 98.25, 97.64 ]
# x = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# y = [98.31, 98.35, 98.37, 98.39, 98.39, 98.41, 98.42, 98.38, 98.37, 98.37, 98.36]
# "r" 表示红色，ms用来设置*的大小
x = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000]
y = [0.00, 40.70, 57.74, 82.38, 89.52, 91.73, 93.63, 94.75,  96.87, 97.35, 97.40, 98.41, 98.42, 98.42]
y1 = [0.00, 32.84, 48.57, 70.35, 79.73, 82.82, 86.21, 88.66,  90.73, 92.19, 93.55, 93.63, 93.72, 93.73]
# 画折线函数
plt.plot(x, y, "blue", marker='*', ms=10, label="TSR-LeNet")
plt.plot(x, y1, "green", linestyle ='-.', ms=10, label="LeNet-5")
# 横坐标刻度旋转
plt.xticks(rotation=45)
plt.xlabel("迭代次数")
plt.ylabel("准确率（%）")
plt.title("模型改进前后准确率对比")
# upper left 将图例a显示到左上角
plt.legend(loc="upper left")

ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为1的倍数

#把y轴的主刻度设置为10的倍数
plt.xlim(0,13000)

# 在折线图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式
for x1, y_1 in zip(x, y):
    plt.text(x1, y_1 + 1, str(y_1), ha='center', va='bottom', fontsize=10, rotation=0)
for x1, y in zip(x, y1):
    plt.text(x1, y - 7, str(y), ha='center', va='bottom', fontsize=10, rotation=0)
plt.show()
