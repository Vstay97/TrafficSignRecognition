# The German Traffic Sign Recognition Benchmark
#
# sample code for reading the traffic sign images and the
# corresponding labels
#
# example:
#            
# trainImages, trainLabels = readTrafficSigns('GTSRB/Training')
# print len(trainLabels), len(trainImages)
# plt.imshow(trainImages[42])
# plt.show()
#
# have fun, Christian

import cv2
import pandas as pd
import numpy as np
from data_augment import create_data
import random
import os
import shutil
# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
def get_data(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    print('数据读取开始')
    for c in range(0,43):
        # 图像路径
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        # gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        # gtReader = pd.read_csv(gtFile, delimiter=';') # csv parser for annotations file

        # gtReader.next() # skip header
        # loop over all images in current annotations file
        # 获得文件夹下所有文件名；filelist为每个遍历结果下各个文件的文件名
        fileList = os.listdir(prefix)
        for i in range(len(fileList)):
            try:
                # 读取图像
                img = cv2.imread(prefix + fileList[i])
                # 图像尺寸缩放
                img = cv2.resize(img, (42, 48))
                # 数据增强
                data_list = create_data(img)
                images.append(img) # the 1th column is the filename
                labels.append(c)
                for data in data_list:
                    # print(data.shape)
                    images.append(data)
                    labels.append(c)
            except Exception as e:
                pass

        # the 8th column is the label
        # gtFile.close()
    # 列表转numpy矩阵
    images = np.asarray(images)
    # 图像的大小用shape属性来获取；分别对应：高、宽、通道数
    print(images.shape)
    print('数据读取完毕')
    return images, labels

def create_test(path0):
    outPath0 = 'test/'
    if not os.path.exists(outPath0):
        os.mkdir(outPath0)
    dir0 = os.listdir(path0)
    print(len(dir0))

    for i in range(len(dir0)):
        img_path = os.path.join(path0, dir0[i])
        imgFile = os.listdir(img_path)
        random.shuffle(imgFile)
        output = os.path.join(outPath0, dir0[i])
        if not os.path.exists(output):
            os.mkdir(output)
        for j in range(round(len(imgFile) * 0.2)):
            shutil.move(img_path + '/' + imgFile[j], output + '/' + imgFile[j])
    # print(round(len(testPath0) * 0.2), len(testPath0))
    # for i in range(round(len(dir0) * 0.2)):




if __name__ == '__main__':
    create_test('GTSRB/Final_Training/Images/')
    # trainImages, trainLabels = get_data('GTSRB/Final_Training/Images')
    # print(len(trainLabels), len(trainImages))
    # cv2.imshow('1', trainImages[42])
    # cv2.waitKey(0)