from cnn_model import get_model
import numpy as np
import keras
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from read_img import get_data


def convert2oneHot(index, Lens):
    hot = np.zeros((Lens,))
    hot[int(index)] = 1
    return hot

# 获取图像数据和标签
data, label = get_data('GTSRB/Final_Training/Images')

# 标签onehot编码
print(label)
labelOneHot = np.array([convert2oneHot(l, 43) for l in label])
print(labelOneHot)
# 划分训练集和验证集（x代表图像数据，y代表标签）
x_train, x_test, y_train, y_test = train_test_split(data, labelOneHot, test_size=0.2)
# 图像数据的归一化
x_train = x_train / 255
x_test = x_test / 255
# print(x_train.shape)
# 定义输入数据的形状
input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
# print(input_shape)

# 神经网络参数配置
npClass = 43
# 调参（超参数）
## 随机失活率
dropout = 0.5
## 学习率
lr = 0.0001
## 批量
batch_size = 8
## 训练集迭代轮数
epochs = 100
# 获取LeNet卷积神经网络模型
model = get_model(input_shape)
# 定义优化函数（Adam）
opt = keras.optimizers.Adam(lr=lr, epsilon=1e-06)
# 模型编译，定义了优化函数、损失函数：多分类交叉熵，还有评价指标
model.compile(optimizer=opt,loss='categorical_crossentropy', metrics=['accuracy'])
# 训练模型， validation_data：验证集，
history = model.fit(x_train,y_train,batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test),verbose=1,callbacks=[TensorBoard(log_dir='./log1')])
# 保存模型
model.save('origin_model.h5')
