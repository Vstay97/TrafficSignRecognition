import cv2
import os
from keras.models import load_model
import numpy as np
import datetime
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd

model = load_model('origin_model.h5')


rootpath = 'test/'


# gtFile = open(prefix + 'GT-final_test.test.csv') # annotations file
# gtReader = pd.read_csv(gtFile, delimiter=';') # csv parser for annotations file
# print(gtReader)
images = []
labels = []
for c in range(0, 43):
    prefix = rootpath + '/' + format(c, '05d') + '/'  # subdirectory for class
    # gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')  # annotations file
    # gtReader = pd.read_csv(gtFile, delimiter=';')  # csv parser for annotations file
    fileList = os.listdir(prefix)
    # gtReader.next() # skip header
    # loop over all images in current annotations file
    for i in range(len(fileList)):
        try:
            # 读取文件，然后缩放
            img = cv2.imread(prefix + fileList[i])
            img = cv2.resize(img, (42, 48))
            # 给测试集的图片加上对应的标签
            images.append(img)  # the 1th column is the filename
            labels.append(c)
        except:
            pass
# 把images列表准换成数组
test_data = np.asarray(images)
# 图像归一化
test_data = test_data / 255
# 预测出的输出数组
y_pred = []


for i in range(len(test_data)):
    # 因为keras要求是四维的数据，所以需要在最前面添加一个维度
    img = np.expand_dims(test_data[i], axis=0)
    # 根据图像预测出类别，结果是一个numpy数组。把他转换成list
    label = list(model.predict(img)[0])
    # 选一个最大值
    l = label.index(max(label))
    # 输出预测的类别
    #------ print(l)
    y_pred.append(l)

# 把之前labels列表转换成数组，因为accuracy score只接受数组
y_true = np.asarray(labels)
print(y_pred)
# 根据预测值和真实值计算准确率
acc = accuracy_score(y_true, y_pred)
print(acc)
