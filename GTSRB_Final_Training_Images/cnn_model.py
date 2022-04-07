from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, GlobalAveragePooling2D,MaxPool2D
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, Activation, Convolution2D, add
from keras.models import Model
from keras.layers import Input, concatenate
from keras import Sequential


def create_model(inputs):
    """

    :param inputs: 训练图像形状
    :return: 模型对象
    """
    model = Sequential()
    # layer_1
    # Conv2D：卷积层， filters:卷积核数量 kernel_size:卷积核大小 activation:激活函数
    model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='valid', input_shape=inputs, activation='tanh'))
    # MaxPooling2D：最大池化层， pool_size：池化核大小
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(5, 5), padding='valid', activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(5, 5), padding='valid', activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # FLatten：展开
    ## [[1,2], [3,4]], flatten [1,2,3,4]
    model.add(Flatten())
    # Dense全连接层， 1200代表全连接层神经元数量
    model.add(Dense(1200, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(840, activation='tanh'))
    # 最后一个全连接层神经元数量与类别数相同
    model.add(Dense(43, activation='softmax'))

    return model

def get_model(shape):

    model = create_model(shape)
    # 模型总结
    model.summary()
    return model
