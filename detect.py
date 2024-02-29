
import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
# 读取图像
def segment_plate_chars(image_path):
    # 读取图像
    img = cv2.imread(image_path)

    # 图像预处理
    img1 = cv2.resize(img, (320, 100), interpolation=cv2.INTER_AREA)
    img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img3 = cv2.bilateralFilter(img2, 11, 17, 17)
    img4 = cv2.Canny(img3, 50, 150)
    img5 = img4[10:90, 10:310]
    crop_img = img1[10:90, 10:310, :]

    # 查找轮廓
    contours, _ = cv2.findContours(img5, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(crop_img, contours, -1, (0, 255, 0), 3)
    # cv2.imshow('contours', crop_img)
    # cv2.waitKey(0)
    candidate = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 500 < w*h < 4000 and 20 < h < 80 and w < 80:
            candidate.append([x, (x + w)])
    # 基于轮廓位置创建一个标记数组
    loc = np.zeros(300)
    for j in range(len(candidate)):
        x1, x2 = candidate[j]
        loc[x1:x2] = 1
    
    # 查找字符的开始和结束位置
    start, end = [], []
    if loc[0] == 1:
        start.append(0)
    for j in range(299):
        if loc[j] == 0 and loc[j+1] == 1:
            start.append(j+1)
        if loc[j] == 1 and loc[j+1] == 0:
            end.append(j)
    if loc[299] == 1:
        end.append(299)

    # 分割字符
    char_images = []
    print(len(start), len(end))
    if (len(start) == 8 and len(end) == 8) or True:
        for j in range(len(start)):
            x1, x2 = start[j], end[j]
            y1, y2 = 0, 80
            char_img = crop_img[y1:y2, x1:x2]
            char_images.append(char_img)
    return char_images

def resize_with_padding(img, target_width, target_height, target_value=255):
    # 计算原始图像的宽高比
    h, w = img.shape[:2]
    ratio = w / h

    # 计算目标宽高比
    target_ratio = target_width / target_height

    # 根据宽高比决定是填充还是缩放
    if ratio > target_ratio:
        # 基于宽度进行缩放
        new_width = target_width
        new_height = int(new_width / ratio)
    else:
        # 基于高度进行缩放
        new_height = target_height
        new_width = int(new_height * ratio)

    # 缩放图像
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 创建一个新的目标尺寸的空白图像
    padded_img = np.full((target_height, target_width), target_value, dtype=np.uint8)


    # 计算填充边界
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # 将缩放后的图像复制到空白图像的中心
    padded_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_img

    return padded_img