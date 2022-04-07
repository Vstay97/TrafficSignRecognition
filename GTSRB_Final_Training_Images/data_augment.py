import cv2
import numpy as np


# 数据增强
def create_data(img):
    # print(img_change.shape)
    ## 保存图像
    # cv2.imwrite('./data_argumen/nosmoke/pingyi/%s' % nosmokeFile[i], img_change)
    # 图像水平翻转
    h_flip = cv2.flip(img, 1)
    # cv2.imwrite('./data_argumen/nosmoke/hflip/%s' % nosmokeFile[i], h_flip)
    # 图像垂直翻转
    v_flip = cv2.flip(img, 0)
    # # cv2.imwrite('./data_argumen/nosmoke/vflip/%s' % nosmokeFile[i], v_flip)
    # # 图像水平垂直翻转
    # hv_flip = cv2.flip(img, -1)
    # cv2.imwrite('./data_argumen/nosmoke/hvflip/%s' % nosmokeFile[i], hv_flip)
    # img.shape的返回值为：高、宽、通道数；这里分别把高度和宽度赋值给rows和cols
    rows, cols = img.shape[:2]
    # # 图像30度旋转
    # M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 30, 1)
    # dst = cv2.warpAffine(img, M, (cols, rows))
    # cv2.imwrite('./data_argumen/nosmoke/30duxz/%s' % nosmokeFile[i], dst)
    # 图像60度旋转
    # M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 60, 1)
    # dst1 = cv2.warpAffine(img, M, (cols, rows))
    # cv2.imwrite('./data_argumen/nosmoke/60duxz/%s' % nosmokeFile[i], dst1)
    # 仿射变换
    point1 = np.float32([[50, 50], [300, 50], [50, 200]])
    point2 = np.float32([[10, 100], [300, 50], [100, 250]])
    M = cv2.getAffineTransform(point1, point2)
    dst2 = cv2.warpAffine(img, M, (cols, rows), borderValue=(0, 0, 0))
    # cv2.imwrite('./data_argumen/nosmoke/fangshe/%s' % nosmokeFile[i], dst2)
    # 图像颜色变换

    # 彩色图像均衡化,需要分解通道 对每一个通道均衡化
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    # 合并每一个通道
    result = cv2.merge((bH, gH, rH))
    return [h_flip, v_flip, dst2, result]

