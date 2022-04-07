import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
# 构建数据
# x = ['Max-pooling', 'Mean-pooling', 'Stochastic-pooling']
# y1 = [95.35, 92.35, 93.16]
# y2 = [93.41, 91.28, 92.30]
# x = ['dropout=0.4', 'dropout=0.5', 'dropout=0.6', 'dropout=0.7', 'no-dropout']
# y1 = [95.37, 96.42, 95.39, 94.78, 95.35]
# y2 = [94.22, 95.31, 94.8, 92.15, 93.40]
x = ['原图像', '预处理图像']
y1 = [96.42, 97.69]
y2 = [95.31, 96.81]
bar_width = 0.2
# 将X轴数据改为使用range(len(x_data), 就是0、1、2...
plt.bar(x=range(len(x)), height=y1, label='训练准确率（%）', color='blue', alpha=0.8, width=bar_width)
# 将X轴数据改为使用np.arange(len(x_data))+bar_width,
# 就是bar_width、1+bar_width、2+bar_width...这样就和第一个柱状图并列了
plt.bar(x=np.arange(len(x)) + bar_width, height=y2, label='测试准确率（%）', color='indianred', alpha=0.8, width=bar_width)
# 在柱状图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式
for x1, yy in enumerate(y1):
    plt.text(x1, yy + 1, str(yy), ha='center', va='bottom', fontsize=10, rotation=0)
for x1, yy in enumerate(y2):
    plt.text(x1 + bar_width, yy + 1, str(yy), ha='center', va='bottom', fontsize=10, rotation=0)
# 设置标题
plt.title("原图像与预处理图像识别准确率对比")
# 设置字符串刻度
# index_ls = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
_ = plt.xticks([0.15, 1.1], x)
# 为两条坐标轴设置名称

plt.xlabel("图像类型")
plt.ylabel("准确率（%）")
# 显示图例
plt.legend(loc="right")
plt.show()
