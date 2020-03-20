# --------------------------------------------------------
# Tensorflow TIN
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ult.config import cfg
from ult.timer import Timer
from ult.ult import Get_next_sp_with_pose,Generate_part_bbox

import cv2
import pickle
import numpy as np
import os
import sys
import glob
import time
import ipdb

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

human_num_thres = 4
object_num_thres = 4

def get_blob(image_id):
    im_file  = cfg.DATA_DIR + '/' + 'hico_20160224_det/images/test2015/HICO_test2015_' + (str(image_id)).zfill(8) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)
    return im_orig, im_shape

def im_detect(sess, net, image_id, Test_RCNN, object_thres, human_thres, detection):

    # save image information
    This_image = []

    im_orig, im_shape = get_blob(image_id) 
    blobs = {}
    blobs['H_num']       = 1

    blobs['S_boxes'] = np.array([0, 0, 0, im_shape[1] - 1, im_shape[0] - 1]).reshape(1, 5).astype(np.float64)  # 这是增加部分
    # print("************************************************************************")
    print('image_id: ', image_id)
    # print("im_shape: ", im_shape[0], im_shape[1])

    # 构建 Dense HOI Graph, 符合条件的H-O对(这里的O可以是Human)，输入进网络得到一组预测，存入This_image列表中
    for Human_out in Test_RCNN[image_id]:
        if (np.max(Human_out[5]) > human_thres) and (Human_out[1] == 'Human'): # This is a valid human
            blobs['H_boxes'] = np.array([0, Human_out[2][0],  Human_out[2][1],  Human_out[2][2],  Human_out[2][3]]).reshape(1,5)

            # 以下为修改部分
            if (len(Human_out)>=8):
                blobs['P_boxes'] = Generate_part_bbox(Human_out[7], Human_out[2])
            else:
                blobs['P_boxes'] = Generate_part_bbox(None, Human_out[2])

            for Object in Test_RCNN[image_id]:
                # 1.the object detection result should > thres  2.the bbox detected is not an object
                # Dense HOI Graph里的 valid Human会和除自身以外所有符合置信度大于object_thres的物体（包括人在内）连线
                if (np.max(Object[5]) > object_thres) and not (np.all(Object[2] == Human_out[2])): # This is a valid object
                    blobs['O_boxes'] = np.array([0, Object[2][0],  Object[2][1],  Object[2][2],  Object[2][3]]).reshape(1,5)
                    blobs['sp']      = Get_next_sp_with_pose(Human_out[2], Object[2], Human_out[6]).reshape(1, 64, 64, 3)

                    # prediction_HO, prediction_PP: 形状为：[1,600]    第一维度为batch_size,test时始终为1，第二维度是600个HOI标签关系的预测
                    # prediction_binary: 形状为: [1,2]  第一维度为batch_size,test时始终为1，第二维度是有无HOI关系的预测
                    prediction_TIN, prediction_binary, prediction_PP, prediction_H,prediction_O,prediction_sp = net.test_image_HO(sess, im_orig, blobs)

                    temp = []
                    temp.append(Human_out[2])           # Human box
                    temp.append(Object[2])              # Object box
                    temp.append(Object[4])              # Object class
                    temp.append(prediction_TIN[0])       # Score (600)
                    temp.append(Human_out[5])           # Human score
                    temp.append(Object[5])              # Object score
                    temp.append(prediction_binary[0])   # binary score
                    temp.append(prediction_PP[0])
                    # temp.append(prediction_H[0])
                    # temp.append(prediction_O[0])
                    # temp.append(prediction_sp[0])
                    # print("########################################################################################")
                    # print("prediction_TIN[0] is: ",prediction_TIN[0][:60])
                    # print("prediction_PP[0] is: ",prediction_PP[0][:60])
                    # print("prediction_H[0] is: ", prediction_H[0][:60])
                    # print("prediction_O[0] is: ", prediction_O[0][:60])
                    # print("prediction_sp[0] is: ", prediction_sp[0][:60])
                    This_image.append(temp)

    # 当 Dense HOI Graph 没有valid的连线时，忽视 human_thres和 object_thres, 采样不大于4个H-O对
    # 与之前不同的是，这里O不能是Human
    if len(This_image) == 0:
        # print("************************************************************************")
        print('Dealing with zero-sample test Image '+str(image_id))

        list_human_included = []
        list_object_included = []
        Human_out_list = []
        Object_list = []

        test_pair_all = Test_RCNN[image_id]
        length = len(test_pair_all)

        while (len(list_human_included) < human_num_thres) or (len(list_object_included) < object_num_thres):
            h_max = [-1, -1.0]
            o_max = [-1, -1.0]
            flag_continue_searching = 0
            for i in range(length):
                if test_pair_all[i][1] == 'Human':
                    if (np.max(test_pair_all[i][5]) > h_max[1]) and not (i in list_human_included) and len(list_human_included) < human_num_thres:
                        h_max = [i, np.max(test_pair_all[i][5])]
                        flag_continue_searching = 1
                else:
                    if np.max(test_pair_all[i][5]) > o_max[1] and not (i in list_object_included) and len(list_object_included) < object_num_thres:
                        o_max = [i, np.max(test_pair_all[i][5])]
                        flag_continue_searching = 1

            if flag_continue_searching == 0:
                break

            list_human_included.append(h_max[0])
            list_object_included.append(o_max[0])

            Human_out_list.append(test_pair_all[h_max[0]])
            Object_list.append(test_pair_all[o_max[0]])


        for Human_out in Human_out_list:
            for Object in Object_list:
                blobs['H_boxes'] = np.array([0, Human_out[2][0],  Human_out[2][1],  Human_out[2][2],  Human_out[2][3]]).reshape(1,5)
                blobs['O_boxes'] = np.array([0, Object[2][0],  Object[2][1],  Object[2][2],  Object[2][3]]).reshape(1,5)
                blobs['sp']      = Get_next_sp_with_pose(Human_out[2], Object[2], Human_out[6]).reshape(1, 64, 64, 3)

                if (len(Human_out) >= 8):
                    blobs['P_boxes'] = Generate_part_bbox(Human_out[7], Human_out[2])
                else:
                    blobs['P_boxes'] = Generate_part_bbox(None, Human_out[2])
                    
                # prediction_HO, prediction_binary = net.test_image_HO(sess, im_orig, blobs)
                prediction_TIN, prediction_binary, prediction_PP, prediction_H,prediction_O,prediction_sp = net.test_image_HO(sess, im_orig, blobs)

                #This_image = []

                temp = []
                temp.append(Human_out[2])           # Human box
                temp.append(Object[2])              # Object box
                temp.append(Object[4])              # Object class
                temp.append(prediction_TIN[0])       # Score (600)
                temp.append(Human_out[5])           # Human score
                temp.append(Object[5])              # Object score
                temp.append(prediction_binary[0])   # binary score
                temp.append(prediction_PP[0])
                # print("prediction_PP[0] is: ", prediction_PP[0])
                This_image.append(temp)
            
    detection[image_id] = This_image


'''
def im_detect_remaining(sess, net, image_id, Test_RCNN, object_thres, human_thres, detection):

    # for those pairs which do not have any pairs with given threshold
    # save image information
    This_image = []

    im_orig, im_shape = get_blob(image_id)

    num_inter = 0
    num_no_inter = 0
    
    blobs = {}
    blobs['H_num']       = 1

    if image_id not in all_remaining:
        return 0, 0
    
    h_max = [-1, -1.0]
    o_max = [-1, -1.0]

    test_pair_all = Test_RCNN[image_id]

    length = len(test_pair_all)
    for i in range(length):
        if test_pair_all[i][1] == 'Human':
            if np.max(test_pair_all[i][5]) > h_max[1]:
                h_max = [i, np.max(test_pair_all[i][5])]
        else:
            if np.max(test_pair_all[i][5]) > o_max[1]:
                o_max = [i, np.max(test_pair_all[i][5])]

    Human_out = test_pair_all[h_max[0]]
    Object = test_pair_all[o_max[0]]
            
    blobs['H_boxes'] = np.array([0, Human_out[2][0],  Human_out[2][1],  Human_out[2][2],  Human_out[2][3]]).reshape(1,5)
    blobs['O_boxes'] = np.array([0, Object[2][0],  Object[2][1],  Object[2][2],  Object[2][3]]).reshape(1,5)
    blobs['sp']      = Get_next_sp_with_pose(Human_out[2], Object[2], Human_out[6]).reshape(1, 64, 64, 3)
                    
    prediction_HO, prediction_binary = net.test_image_HO(sess, im_orig, blobs)

    This_image = []

    temp = []
    temp.append(Human_out[2])           # Human box
    temp.append(Object[2])              # Object box
    temp.append(Object[4])              # Object class
    temp.append(prediction_HO[0])       # Score (600)
    temp.append(Human_out[5])           # Human score
    temp.append(Object[5])              # Object score
    temp.append(prediction_binary[0])   # binary score

    This_image.append(temp)
            
    detection[image_id] = This_image

    return num_inter, num_no_inter
'''

def test_net(sess, net, Test_RCNN, output_dir, object_thres, human_thres):

    np.random.seed(cfg.RNG_SEED)
    detection = {}
    count = 0
    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    # glob.iglob函数获取一个可遍历对象，使用它可以逐个获取匹配的文件路径名
    for line in glob.iglob(cfg.DATA_DIR + '/' + 'hico_20160224_det/images/test2015/*.jpg'):

        _t['im_detect'].tic()
 
        image_id   = int(line[-9:-4])

        #if image_id in all_remaining:
        #    im_detect_remaining(sess, net, image_id, Test_RCNN, object_thres, human_thres, detection)
        #    print('dealing with remaining image')
        #else:
        #    im_detect(sess, net, image_id, Test_RCNN, object_thres, human_thres, detection)
        im_detect(sess, net, image_id, Test_RCNN, object_thres, human_thres, detection)

        _t['im_detect'].toc()
        print('im_detect: {:d}/{:d} {:.3f}s'.format(count + 1, 9658, _t['im_detect'].average_time))
        count += 1

    # 把 detection的结果存入pickle文件中
    pickle.dump( detection, open( output_dir, "wb" ) )
