# --------------------------------------------------------
# Tensorflow TIN
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""
Generating training instance
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json
import pickle
import random
from random import randint
import tensorflow as tf
import cv2
import os.path as osp

#from config import cfg # cause error because of "from __future__ import absolute_import"
from ult import config # use absolute import and config.cfg


list_no_inter = [10,24,31,46,54,65,76,86,92,96,107,111,
129,146,160,170,174,186,194,198,208,214,224,232,235,239,
243,247,252,257,264,273,283,290,295,305,313,325,330,336,
342,348,352,356,363,368,376,383,389,393,397,407,414,418,
429,434,438,445,449,453,463,474,483,488,502,506,516,528,
533,538,546,550,558,562,567,576,584,588,595,600]


OBJECT_MASK = [
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1],
    [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0],
    [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0]
]


print("*************data path:*************")
print(config.cfg.DATA_DIR)
print("************************************")


# 输入原图中 Human box 和 Object box 框的坐标
# 保持 Human 和 Object 框的空间关系与比例不发生变化，将长边放缩至 size
# 包住 H-O pair的框在size*size的pattern特征图的中心位置，如果高比宽长，则水平居中，如果宽比高长，则竖直居中
# 返回 Human box, Object box 在 size*size 的 pattern 特征图中的坐标，两个坐标都是形如[xmin,ymin,xmax,ymax]的含四个元素的一维数组
def bbox_trans(human_box_ori, object_box_ori, size = 64):
    human_box  = human_box_ori.copy()
    object_box = object_box_ori.copy()

    # 提取pattern的特征图是包括一对H-O pair的最小的方框
    InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]), max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
    height = InteractionPattern[3] - InteractionPattern[1] + 1
    width = InteractionPattern[2] - InteractionPattern[0] + 1
    
    if height > width:
        ratio = 'height'
    else:
        ratio = 'width'
        
    # 将新的pattern特征图左上角对齐至 (0,0)
    human_box[0] -= InteractionPattern[0]
    human_box[2] -= InteractionPattern[0]
    human_box[1] -= InteractionPattern[1]
    human_box[3] -= InteractionPattern[1]    
    object_box[0] -= InteractionPattern[0]
    object_box[2] -= InteractionPattern[0]
    object_box[1] -= InteractionPattern[1]
    object_box[3] -= InteractionPattern[1] 

    if ratio == 'height':   # 高大于宽
        # 将同时包住 H-O pair的最小的那个框等比例缩放，使得长为64
        # 得到的框左上角对齐至（0,0），即沿左上方对齐
        human_box[0] = 0 + size * human_box[0] / height
        human_box[1] = 0 + size * human_box[1] / height
        human_box[2] = (size * width / height - 1) - size * (width  - 1 - human_box[2]) / height
        human_box[3] = (size - 1)                  - size * (height - 1 - human_box[3]) / height
        object_box[0] = 0 + size * object_box[0] / height
        object_box[1] = 0 + size * object_box[1] / height
        object_box[2] = (size * width / height - 1) - size * (width  - 1 - object_box[2]) / height
        object_box[3] = (size - 1)                  - size * (height - 1 - object_box[3]) / height
        # 还需要水平移动
        InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]), max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]

        #assert (InteractionPattern[0] == 0) & (InteractionPattern[1] == 0) & (InteractionPattern[3] == 63) & (InteractionPattern[2] <= 63)
        # 包住 H-O pair的框，底边对齐为 size-1
        if human_box[3] > object_box[3]:
            human_box[3] = size - 1
        else:
            object_box[3] = size - 1

        # 向右水平移动，使包住 H-O pair 的框水平对齐至中间
        shift = size / 2 - (InteractionPattern[2] + 1) / 2 
        human_box += [shift, 0 , shift, 0]
        object_box += [shift, 0 , shift, 0]
     
    else:   # 宽大于高
        # 将同时包住 H-O pair的最小的那个框等比例缩放，使得宽为64
        # 得到的框左上角对齐至（0,0），即沿左上方对齐
        human_box[0] = 0 + size * human_box[0] / width
        human_box[1] = 0 + size * human_box[1] / width
        human_box[2] = (size - 1)                  - size * (width  - 1 - human_box[2]) / width
        human_box[3] = (size * height / width - 1) - size * (height - 1 - human_box[3]) / width
        object_box[0] = 0 + size * object_box[0] / width
        object_box[1] = 0 + size * object_box[1] / width
        object_box[2] = (size - 1)                  - size * (width  - 1 - object_box[2]) / width
        object_box[3] = (size * height / width - 1) - size * (height - 1 - object_box[3]) / width
        # 还需要竖直移动
        InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]), max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]

        #assert (InteractionPattern[0] == 0) & (InteractionPattern[1] == 0) & (InteractionPattern[2] == 63) & (InteractionPattern[3] <= 63)
        # 包住 H-O pair的框，右边对齐为 size-1
        if human_box[2] > object_box[2]:
            human_box[2] = size - 1
        else:
            object_box[2] = size - 1

        # 向下竖直移动，使包住 H-O pair 的框竖直对齐至中间
        shift = size / 2 - (InteractionPattern[3] + 1) / 2
        human_box = human_box + [0, shift, 0 , shift]
        object_box = object_box + [0, shift, 0 , shift]

    return np.round(human_box), np.round(object_box)


def Get_next_sp(human_box, object_box):
    
    InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]), max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
    height = InteractionPattern[3] - InteractionPattern[1] + 1
    width = InteractionPattern[2] - InteractionPattern[0] + 1
    if height > width:
        H, O = bbox_trans(human_box, object_box, 'height')
    else:
        H, O  = bbox_trans(human_box, object_box, 'width')
    
    Pattern = np.zeros((64,64,2))
    Pattern[int(H[1]):int(H[3]) + 1,int(H[0]):int(H[2]) + 1,0] = 1
    Pattern[int(O[1]):int(O[3]) + 1,int(O[0]):int(O[2]) + 1,1] = 1


    return Pattern


# 获取两个形如[xmin,ymin,xmax,ymax]的bbox的IOU
def bb_IOU(boxA, boxB):

    ixmin = np.maximum(boxA[0], boxB[0])
    iymin = np.maximum(boxA[1], boxB[1])
    ixmax = np.minimum(boxA[2], boxB[2])
    iymax = np.minimum(boxA[3], boxB[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((boxB[2] - boxB[0] + 1.) * (boxB[3] - boxB[1] + 1.) +
           (boxA[2] - boxA[0] + 1.) *
           (boxA[3] - boxA[1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps


# 对原来的box，产生augment数量的新的box，将新的box添加到原来box的后面，形成二维数组，二维数组的每一维是一个box
# 新的box是由原来box横纵坐标做相应变换得来，和原来box的iou大于threshold(0.7)
# 新的box和原来box四个坐标前都加上0，每个box是长度是5
def Augmented_box(bbox, shape, image_id, augment = 15, break_flag = True):

    thres_ = 0.7

    box = np.array([0, bbox[0],  bbox[1],  bbox[2],  bbox[3]]).reshape(1,5)
    box = box.astype(np.float64)
        
    count = 0
    time_count = 0
    while count < augment:
        
        time_count += 1
        height = bbox[3] - bbox[1]
        width  = bbox[2] - bbox[0]

        height_cen = (bbox[3] + bbox[1]) / 2
        width_cen  = (bbox[2] + bbox[0]) / 2

        ratio = 1 + randint(-10,10) * 0.01

        height_shift = randint(-np.floor(height),np.floor(height)) * 0.1
        width_shift  = randint(-np.floor(width),np.floor(width)) * 0.1

        H_0 = max(0, width_cen + width_shift - ratio * width / 2)
        H_2 = min(shape[1] - 1, width_cen + width_shift + ratio * width / 2)
        H_1 = max(0, height_cen + height_shift - ratio * height / 2)
        H_3 = min(shape[0] - 1, height_cen + height_shift + ratio * height / 2)
        
        
        if bb_IOU(bbox, np.array([H_0, H_1, H_2, H_3])) > thres_:
            box_ = np.array([0, H_0, H_1, H_2, H_3]).reshape(1,5)
            box  = np.concatenate((box,     box_),     axis=0)
            count += 1
        if break_flag == True and time_count > 150:
            return box
            
    return box


def Generate_action(action_list):
    action_ = np.zeros(29)
    for GT_idx in action_list:
        action_[GT_idx] = 1
    action_ = action_.reshape(1,29)
    return action_


# 返回一个[1,600]大小的二维数组，数组里面为1的项，是存在的HOI label（可能不止一个）
def Generate_action_HICO(action_list):
    action_ = np.zeros(600)
    for GT_idx in action_list:
        action_[GT_idx] = 1
    action_ = action_.reshape(1,600)
    return action_


##############################################################################
##############################################################################
# tools added
##############################################################################
##############################################################################

def Generate_action_30(action_list):
    action_ = np.zeros(30)
    for GT_idx in action_list:
        action_[GT_idx] = 1
    action_ = action_.reshape(1,30)
    return action_


# 返回 size*size 大小, 将相关关节用不同深浅(0.15-0.95)的直线连接在一起的 Pose Map
# 除了连线的线条上的像素，其余像素值均为 0
def draw_relation(human_pattern, joints, size = 64):

    joint_relation = [[1,3],[2,4],[0,1],[0,2],[0,17],[5,17],[6,17],[5,7],[6,8],[7,9],[8,10],[11,17],[12,17],[11,13],[12,14],[13,15],[14,16]]
    color = [0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
    skeleton = np.zeros((size, size, 1), dtype="float32")

    for i in range(len(joint_relation)):
        cv2.line(skeleton, tuple(joints[joint_relation[i][0]]), tuple(joints[joint_relation[i][1]]), (color[i]))
    
    # cv2.rectangle(skeleton, (int(human_pattern[0]), int(human_pattern[1])), (int(human_pattern[2]), int(human_pattern[3])), (255))
    # cv2.imshow("Joints", skeleton)
    # cv2.waitKey(0)
    # print(skeleton[:,:,0])

    return skeleton


# 将读取的 pose 的关键点的坐标信息变换到 size*size 图上，记为 joints
# 调用函数draw_relation() ，输入 size*size 图上的 joints （的坐标信息）
# 返回 size*size 大小, 将相关关节用不同深浅(0.15-0.95)的直线连接在一起的 Pose Map
def get_skeleton(human_box, human_pose, human_pattern, num_joints = 17, size = 64):
    width = human_box[2] - human_box[0] + 1
    height = human_box[3] - human_box[1] + 1
    pattern_width = human_pattern[2] - human_pattern[0] + 1
    pattern_height = human_pattern[3] - human_pattern[1] + 1
    joints = np.zeros((num_joints + 1, 2), dtype='int32')

    for i in range(num_joints):
        joint_x, joint_y, joint_score = human_pose[3 * i : 3 * (i + 1)]
        x_ratio = (joint_x - human_box[0]) / float(width)
        y_ratio = (joint_y - human_box[1]) / float(height)
        joints[i][0] = min(size - 1, int(round(x_ratio * pattern_width + human_pattern[0])))
        joints[i][1] = min(size - 1, int(round(y_ratio * pattern_height + human_pattern[1])))
    joints[num_joints] = (joints[5] + joints[6]) / 2     # 取左肩和右肩的中心成为第18个关键点
    '''Result
    for COCO(17 body parts)
    {0, "Nose"},
    {1, "LEye"},
    {2, "REye"},
    {3, "LEar"},
    {4, "REar"},
    {5, "LShoulder"},
    {6, "RShoulder"},
    {7, "LElbow"},
    {8, "RElbow"},
    {9, "LWrist"},
    {10, "RWrist"},
    {11, "LHip"},
    {12, "RHip"},
    {13, "LKnee"},
    {14, "Rknee"},
    {15, "LAnkle"},
    {16, "RAnkle"}'''

    return draw_relation(human_pattern, joints)


# 获取spatial-pose stream，记为 Pattern
# 返回一个 [size, size, 3]的张量
# 三个二维 feature map分别代表 human, object和 human pose
def Get_next_sp_with_pose(human_box, object_box, human_pose, num_joints=17):
    # InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]), max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
    # height = InteractionPattern[3] - InteractionPattern[1] + 1
    # width = InteractionPattern[2] - InteractionPattern[0] + 1
    # if height > width:
    #     H, O = bbox_trans(human_box, object_box, 'height')
    # else:
    #     H, O  = bbox_trans(human_box, object_box, 'width')

    # 提取pattern的特征图是包括一对H-O pair的最小的方框
    H, O = bbox_trans(human_box, object_box)

    # 形成 64*64 pattern图的前两个维度，代表空间位置关系
    # 第一个维度为 Human 框，在 Human 框里的像素点取值为 1，其余位置为 0
    # 第二个维度为 Object 框，在 Object 框里的像素点取值为 1，其余位置为 0
    Pattern = np.zeros((64,64,2), dtype='float32')
    Pattern[int(H[1]):int(H[3]) + 1,int(H[0]):int(H[2]) + 1,0] = 1
    Pattern[int(O[1]):int(O[3]) + 1,int(O[0]):int(O[2]) + 1,1] = 1

    if human_pose != None and len(human_pose) == 51:  # 如果人物姿态存在，则 Pose Map是人物关键点用不同深浅的线条连接在一起的图片
        skeleton = get_skeleton(human_box, human_pose, H, num_joints)
    else:    # 如果人物姿态不存在，则 Pose Map 是人物框内部区域均取深度为 0.05
        skeleton = np.zeros((64,64,1), dtype='float32')
        skeleton[int(H[1]):int(H[3]) + 1,int(H[0]):int(H[2]) + 1,0] = 0.05

    # 将 Spatial Map 和 Pose Map 在最后维度上进行拼接，得到 [size, size, 3]的张量，作为 spatial-pose stream输入到后边的神经网络
    Pattern = np.concatenate((Pattern, skeleton), axis=2)

    return Pattern


def Generate_part_bbox(joint_bbox, Human_bbox=None):
    part_bbox = np.zeros([1, 10, 5], dtype=np.float64)
    if joint_bbox is None or isinstance(joint_bbox, int):
        # print("None!")
        if Human_bbox is None:
            raise ValueError
        for i in range(10):
            part_bbox[0, i, :] = np.array([0, Human_bbox[0], Human_bbox[1], Human_bbox[2], Human_bbox[3]], dtype=np.float64)

    else:
        # print("joint_box: ",joint_bbox)
        for i in range(10):
            part_bbox[0, i, :] = np.array([0, joint_bbox[i]['x1'], joint_bbox[i]['y1'], joint_bbox[i]['x2'], joint_bbox[i]['y2']], dtype=np.float64)

    return part_bbox

##############################################################################
# for vcoco with pose pattern
##############################################################################

def Get_Next_Instance_HO_Neg_pose_pattern_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0][0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H, binary_label = Augmented_HO_Neg_pose_pattern_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes_solo']= Human_augmented_solo
    blobs['H_boxes']     = Human_augmented
    blobs['O_boxes']     = Object_augmented
    blobs['gt_class_HO'] = action_HO
    blobs['gt_class_H']  = action_H
    blobs['Mask_HO']     = mask_HO
    blobs['Mask_H']      = mask_H
    blobs['sp']          = Pattern
    blobs['H_num']       = len(action_H)
    blobs['binary_label'] = binary_label

    return blobs

def Augmented_HO_Neg_pose_pattern_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0][0]
    GT_count = len(GT)
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 

    Human_augmented, Object_augmented, action_HO, Human_augmented_solo, action_H = [], [], [], [], []
    Pattern   = np.empty((0, 64, 64, 3), dtype=np.float32)

    for i in range(GT_count - 1):
        Human    = GT[i][2]
        Object   = GT[i][3]

        Human_augmented_temp  = Augmented_box(Human,  shape, image_id, aug_all)
        Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_all)
        Human_augmented_solo.extend(Human_augmented_temp)

        length_min = min(len(Human_augmented_temp),len(Object_augmented_temp))

        Human_augmented_temp  =  Human_augmented_temp[:length_min]
        Object_augmented_temp = Object_augmented_temp[:length_min]

        action_H__temp = Generate_action(GT[i][4])
        action_H_temp  = action_H__temp
        for j in range(length_min - 1):
            action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)
    
        action_HO__temp = Generate_action(GT[i][1])
        action_HO_temp  = action_HO__temp
        for j in range(length_min - 1):
            action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)

        for j in range(length_min):
            Pattern_ = Get_next_sp_with_pose(Human_augmented_temp[j][1:], Object_augmented_temp[j][1:], GT[i][5]).reshape(1, 64, 64, 3)
            Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

        Human_augmented.extend(Human_augmented_temp)
        Object_augmented.extend(Object_augmented_temp)
        action_HO.extend(action_HO_temp)
        action_H.extend(action_H_temp)

    Human    = GT[GT_count - 1][2]
    Object   = GT[GT_count - 1][3]

    Human_augmented_temp  = Augmented_box(Human,  shape, image_id, aug_last)
    Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_last)
    Human_augmented_solo.extend(Human_augmented_temp)

    length_min = min(len(Human_augmented_temp),len(Object_augmented_temp))

    Human_augmented_temp  =  Human_augmented_temp[:length_min]
    Object_augmented_temp = Object_augmented_temp[:length_min]

    action_H__temp = Generate_action(GT[GT_count - 1][4])
    action_H_temp  = action_H__temp
    for j in range(length_min - 1):
        action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)

    action_HO__temp = Generate_action(GT[GT_count - 1][1])
    action_HO_temp  = action_HO__temp
    for j in range(length_min - 1):
        action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)

    for j in range(length_min):
        Pattern_ = Get_next_sp_with_pose(Human_augmented_temp[j][1:], Object_augmented_temp[j][1:], GT[GT_count - 1][5]).reshape(1, 64, 64, 3)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

    Human_augmented.extend(Human_augmented_temp)
    Object_augmented.extend(Object_augmented_temp)
    action_HO.extend(action_HO_temp)
    action_H.extend(action_H_temp)

    num_pos = len(Human_augmented)

    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                Pattern_ = Get_next_sp_with_pose(np.array(Neg[2], dtype='float64'), np.array(Neg[3], dtype='float64'), Neg[7]).reshape(1, 64, 64, 3)
                Pattern  = np.concatenate((Pattern, Pattern_), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                Pattern_ = Get_next_sp_with_pose(np.array(Neg[2], dtype='float64'), np.array(Neg[3], dtype='float64'), Neg[7]).reshape(1, 64, 64, 3)
                Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

    num_pos_neg = len(Human_augmented)


    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]).reshape(1,29) # some actions do not have objects
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,29)
    mask_HO   = mask_HO_
    mask_H    = mask_H_

    for i in range(num_pos - 1):
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_HO = np.concatenate((action_HO, np.zeros(29).reshape(1,29)), axis=0)

    Human_augmented   = np.array(Human_augmented, dtype = 'float32')
    Human_augmented_solo = np.array(Human_augmented_solo, dtype = 'float32')
    Object_augmented  = np.array(Object_augmented, dtype = 'float32')
    action_HO         = np.array(action_HO, dtype = 'int32') 
    action_H          = np.array(action_H, dtype = 'int32') 


    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 3) 
    Human_augmented   = Human_augmented.reshape( num_pos_neg, 5) 
    Human_augmented_solo = Human_augmented_solo.reshape( -1, 5) 
    Object_augmented  = Object_augmented.reshape(num_pos_neg, 5) 
    action_HO         = action_HO.reshape(num_pos_neg, 29) 
    action_H          = action_H.reshape( num_pos, 29)
    mask_HO           = mask_HO.reshape(  num_pos_neg, 29)
    mask_H            = mask_H.reshape(   num_pos, 29)

    # binary label for vcoco, different from hico, vcoco does not have the no-interaction label, so give it binary label accord to the locations of pos and neg directly
    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    for i in range(num_pos):
        binary_label[i][0] = 1 # pos is at 0
    for i in range(num_pos, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H, binary_label


def Get_Next_Instance_HO_spNeg_pose_pattern_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0][0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label = Augmented_HO_spNeg_pose_pattern_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes']     = Human_augmented
    blobs['Hsp_boxes']   = Human_augmented_sp
    blobs['O_boxes']     = Object_augmented
    blobs['gt_class_sp'] = action_sp
    blobs['gt_class_HO'] = action_HO
    blobs['gt_class_H']  = action_H
    blobs['Mask_sp']     = mask_sp
    blobs['Mask_HO']     = mask_HO
    blobs['Mask_H']      = mask_H
    blobs['sp']          = Pattern
    blobs['H_num']       = len(action_H)
    blobs['binary_label'] = binary_label

    return blobs

def Augmented_HO_spNeg_pose_pattern_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0][0]
    GT_count = len(GT)
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 

    Human_augmented, Object_augmented, action_HO, action_H = [], [], [], []
    Pattern   = np.empty((0, 64, 64, 3), dtype=np.float32)

    for i in range(GT_count - 1):
        Human    = GT[i][2]
        Object   = GT[i][3]

        Human_augmented_temp  = Augmented_box(Human,  shape, image_id, aug_all)
        Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_all)

        length_min = min(len(Human_augmented_temp),len(Object_augmented_temp))

        Human_augmented_temp  =  Human_augmented_temp[:length_min]
        Object_augmented_temp = Object_augmented_temp[:length_min]

        action_H__temp = Generate_action(GT[i][4])
        action_H_temp  = action_H__temp
        for j in range(length_min - 1):
            action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)
    
        action_HO__temp = Generate_action(GT[i][1])
        action_HO_temp  = action_HO__temp
        for j in range(length_min - 1):
            action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)

        for j in range(length_min):
            Pattern_ = Get_next_sp_with_pose(Human_augmented_temp[j][1:], Object_augmented_temp[j][1:], GT[i][5]).reshape(1, 64, 64, 3)
            Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

        Human_augmented.extend(Human_augmented_temp)
        Object_augmented.extend(Object_augmented_temp)
        action_HO.extend(action_HO_temp)
        action_H.extend(action_H_temp)

    Human    = GT[GT_count - 1][2]
    Object   = GT[GT_count - 1][3]

    Human_augmented_temp  = Augmented_box(Human,  shape, image_id, aug_last)
    Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_last)

    length_min = min(len(Human_augmented_temp),len(Object_augmented_temp))

    Human_augmented_temp  =  Human_augmented_temp[:length_min]
    Object_augmented_temp = Object_augmented_temp[:length_min]

    action_H__temp = Generate_action(GT[GT_count - 1][4])
    action_H_temp  = action_H__temp
    for j in range(length_min - 1):
        action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)

    action_HO__temp = Generate_action(GT[GT_count - 1][1])
    action_HO_temp  = action_HO__temp
    for j in range(length_min - 1):
        action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)

    for j in range(length_min):
        Pattern_ = Get_next_sp_with_pose(Human_augmented_temp[j][1:], Object_augmented_temp[j][1:], GT[GT_count - 1][5]).reshape(1, 64, 64, 3)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

    Human_augmented.extend(Human_augmented_temp)
    Object_augmented.extend(Object_augmented_temp)
    action_HO.extend(action_HO_temp)
    action_H.extend(action_H_temp)

    action_sp = np.array(action_HO).copy()

    num_pos = len(Human_augmented)

    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                Pattern_ = Get_next_sp_with_pose(np.array(Neg[2], dtype='float64'), np.array(Neg[3], dtype='float64'), Neg[7]).reshape(1, 64, 64, 3)
                Pattern  = np.concatenate((Pattern, Pattern_), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                Pattern_ = Get_next_sp_with_pose(np.array(Neg[2], dtype='float64'), np.array(Neg[3], dtype='float64'), Neg[7]).reshape(1, 64, 64, 3)
                Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

    num_pos_neg = len(Human_augmented)

    mask_sp_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]).reshape(1,29)
    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]).reshape(1,29)
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,29)
    mask_sp   = mask_sp_
    mask_HO   = mask_HO_
    mask_H    = mask_H_

    for i in range(num_pos - 1):
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_sp   = np.concatenate((mask_sp,   mask_sp_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_sp = np.concatenate((action_sp, np.zeros(29).reshape(1,29)), axis=0) # neg has all 0 in 29 label

    Human_augmented   = np.array(Human_augmented, dtype = 'float32')
    Object_augmented  = np.array(Object_augmented, dtype = 'float32')
    action_HO         = np.array(action_HO, dtype = 'int32') 
    action_sp         = np.array(action_sp, dtype = 'int32') 
    action_H          = np.array(action_H, dtype = 'int32') 


    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 3) 
    Human_augmented_sp= Human_augmented.reshape( num_pos_neg, 5) 
    Human_augmented   = Human_augmented[:num_pos].reshape( num_pos, 5) 
    Object_augmented  = Object_augmented[:num_pos].reshape(num_pos, 5)
    action_sp         = action_sp.reshape(num_pos_neg, 29)
    action_HO         = action_HO.reshape(num_pos, 29) 
    action_H          = action_H.reshape( num_pos, 29)
    mask_sp           = mask_sp.reshape(num_pos_neg, 29)
    mask_HO           = mask_HO.reshape(  num_pos, 29)
    mask_H            = mask_H.reshape(   num_pos, 29)

    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    for i in range(num_pos):
        binary_label[i][0] = 1 # pos is at 0
    for i in range(num_pos, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label


##############################################################################
# for hico with pose map
##############################################################################

# 这里的输入trainval_GT是已经经过change form函数过了的，每一项是一个二维数组，包含同一张图片中至多5个proposal的信息
def Get_Next_Instance_HO_Neg_HICO_pose_pattern_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]   # 包含某一张图片的至多5个ground truth proposal，同一张图片的proposals可能作为几个GT被读取
    image_id = GT[0][0]
    im_file = config.cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + str(image_id).zfill(8) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)        # 将图片的张量改为 (1, height, weight, 3)

    Pattern, Human_augmented, Object_augmented, action_HO, Part_bbox, num_pos, binary_label = Augmented_HO_Neg_HICO_pose_pattern_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)

    # 以下是新增部分
    Image_augmented = [np.empty((0, 5), dtype=np.float64)]
    for i in range(Human_augmented.shape[0]):
        Image_augmented.append(np.array([0, 0, 0, im_shape[1] - 1, im_shape[0] - 1]).reshape(1, 5).astype(np.float64))
    Image_augmented = np.concatenate(Image_augmented, axis=0)

    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes']     = Human_augmented
    blobs['O_boxes']     = Object_augmented
    blobs['gt_class_HO'] = action_HO
    blobs['sp']          = Pattern
    blobs['P_boxes'] = Part_bbox        # partpair 新增项
    blobs['S_boxes'] = Image_augmented  # partpair 新增项
    blobs['H_num']       = num_pos
    blobs['binary_label'] = binary_label

    # print("##################################################################")
    # print("image_id is: ",image_id)
    # print("shape of Human_augmented: ", blobs['H_boxes'].shape)
    # print("shape of P_bbox: ", blobs['P_boxes'].shape)
    # print("blobs['P_boxes']: ",blobs['P_boxes'])
    # print("shape of S_bbox: ", blobs['S_boxes'].shape)
    # print("blobs['S_boxes']: ", blobs['S_boxes'])

    return blobs

# 遗留问题： trainval_Neg是怎么产生的？
# 返回某一张图片的 Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label
# 所有返回值的长度均为 num_pos_neg
def Augmented_HO_Neg_HICO_pose_pattern_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    GT_count = len(GT)    # GT中proposal的个数
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 
    image_id = GT[0][0]

    Human_augmented, Object_augmented, action_HO, part_bbox = [], [], [], []
    Pattern   = np.empty((0, 64, 64, 3), dtype=np.float32)

    # 前 GT-1个H-O pair, 每个会被 augment共aug_all次，加上原来的，一共 aug_all+1 个新的pair
    for i in range(GT_count - 1):
        Human    = GT[i][2]
        Object   = GT[i][3]

        Human_augmented_temp  = Augmented_box(Human,  shape, image_id, aug_all)
        Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_all)
        # 可能采样150次还没有采样到足够的符合条件的框，这时候augment会提前终止，导致增强的框的总数没有到达aug_all
        # 这时候根据length_min来更改Human,Object的框的数目
        length_min = min(len(Human_augmented_temp),len(Object_augmented_temp))
        Human_augmented_temp  =  Human_augmented_temp[:length_min]
        Object_augmented_temp = Object_augmented_temp[:length_min]
    
        action_HO__temp = Generate_action_HICO(GT[i][1])
        action_HO_temp  = action_HO__temp
        part_bbox__temp = Generate_part_bbox(GT[i][7], Human)
        part_bbox_temp = part_bbox__temp
        # shape of action_HO_temp: [length_min,600]，二维数组
        # 该数组每个元素是相同的长度为600的数组，代表 H-O pair之间存在的HOI关系
        for j in range(length_min - 1):
            action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)
            part_bbox_temp = np.concatenate((part_bbox_temp, part_bbox__temp), axis=0)

        for j in range(length_min):
            # 切片从 1 开始是因为 Human(或Object)_augmented_temp 每个box都是 [0, xmin, ymin, xmax, ymax] 这样表示的
            Pattern_ = Get_next_sp_with_pose(Human_augmented_temp[j][1:], Object_augmented_temp[j][1:], GT[i][5]).reshape(1, 64, 64, 3)
            Pattern  = np.concatenate((Pattern, Pattern_), axis=0)   # 形状为 [length_min, 64, 64, 3]

        # list.extend(seq) 将序列添加到原来list的后面，与原来的序列合并
        Human_augmented.extend(Human_augmented_temp)
        Object_augmented.extend(Object_augmented_temp)
        action_HO.extend(action_HO_temp)
        part_bbox.extend(part_bbox_temp)

    # 最后一个 proposal pair, 被augment成aug_last个pair
    Human    = GT[GT_count - 1][2]
    Object   = GT[GT_count - 1][3]

    Human_augmented_temp  = Augmented_box(Human,  shape, image_id, aug_last)
    Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_last)

    length_min = min(len(Human_augmented_temp),len(Object_augmented_temp))
    Human_augmented_temp  =  Human_augmented_temp[:length_min]
    Object_augmented_temp = Object_augmented_temp[:length_min]

    action_HO__temp = Generate_action_HICO(GT[GT_count - 1][1])   # GT[GT_count - 1][1]是 [xx]
    action_HO_temp  = action_HO__temp
    part_bbox__temp = Generate_part_bbox(GT[GT_count - 1][7], Human)
    part_bbox_temp = part_bbox__temp
    for j in range(length_min - 1):
        action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)
        part_bbox_temp = np.concatenate((part_bbox_temp, part_bbox__temp), axis=0)

    for j in range(length_min):
        Pattern_ = Get_next_sp_with_pose(Human_augmented_temp[j][1:], Object_augmented_temp[j][1:], GT[GT_count - 1][5]).reshape(1, 64, 64, 3)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

    Human_augmented.extend(Human_augmented_temp)
    Object_augmented.extend(Object_augmented_temp)
    action_HO.extend(action_HO_temp)
    part_bbox.extend(part_bbox_temp)

    num_pos = len(Human_augmented)   # 数量为 len(GT)+15，即原来的 len(GT)个框，再加上后来的增加的15个框（PoseArgument=15)

    if image_id in Trainval_Neg:   # Trainval_Neg[image_id]包含标号为image_id的这张图片错误的 HOI 对
        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                action_HO        = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)  # [Neg[1]] 是 [xx]
                part_bbox        = np.concatenate((part_bbox, Generate_part_bbox(Neg[8], Neg[2])), axis=0)
                Pattern_ = Get_next_sp_with_pose(np.array(Neg[2], dtype='float64'), np.array(Neg[3], dtype='float64'), Neg[7]).reshape(1, 64, 64, 3)
                Pattern  = np.concatenate((Pattern, Pattern_), axis=0)
        else:
            # List 长度为 len(Trainval_Neg[image_id])，每个元素都在 [0,Trainval_Neg[image_id] )范围里
            # List 的前 Neg_select 项用以提供 Neg H-O 对的索引
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                action_HO        = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)
                part_bbox = np.concatenate((part_bbox, Generate_part_bbox(Neg[8], Neg[2])), axis=0)
                Pattern_ = Get_next_sp_with_pose(np.array(Neg[2], dtype='float64'), np.array(Neg[3], dtype='float64'), Neg[7]).reshape(1, 64, 64, 3)
                Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

    num_pos_neg = len(Human_augmented)

    Pattern           = Pattern.reshape(num_pos_neg, 64, 64, 3) 
    Human_augmented   = np.array(Human_augmented, dtype='float32').reshape(num_pos_neg, 5) 
    Object_augmented  = np.array(Object_augmented, dtype='float32').reshape(num_pos_neg, 5) 
    action_HO         = np.array(action_HO, dtype='int32').reshape(num_pos_neg, 600)
    part_bbox         = np.array(part_bbox, dtype='int32')

    # binary label用以区分是 Positive 的 H-O pair还是 Negative 的 H-O pair
    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    for i in range(num_pos):
        # if hoi_id belong to 80 then is 0 1, else 1 0
        binary_label[i][0] = 1 # pos is at 0 #############problem: here in pos, it contains the no interaction 80, so we may give this 80 '0 1' and 520 '1 0', and test
    for i in range(num_pos, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    # Pattern的形状为： (num_pos_neg, 64, 64, 3)
    return Pattern, Human_augmented, Object_augmented, action_HO, part_bbox, num_pos, binary_label
