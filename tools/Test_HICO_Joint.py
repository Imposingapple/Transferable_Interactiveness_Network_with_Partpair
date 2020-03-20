# --------------------------------------------------------
# Tensorflow TIN
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import tensorflow as tf
import numpy as np
import argparse
import pickle
import json
import ipdb
import os

from networks.TIN_HICO import ResNet50
from ult.config import cfg
from models.test_HICO_pose_pattern_all_wise_pair import test_net

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # use GPU 0

def parse_args():
    parser = argparse.ArgumentParser(description='Test TIN on HICO dataset')
    parser.add_argument('--num_iteration', dest='iteration',
            help='Specify which weight to load',
            default=2000000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='TIN_partpair', type=str)
    parser.add_argument('--object_thres', dest='object_thres',
            help='Object threshold',
            default=0.3, type=float) 
    parser.add_argument('--human_thres', dest='human_thres',
            help='Human threshold',
            default=0.8, type=float)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    num_iter = args.iteration

    test_output = cfg.ROOT_DIR + '/-Results/' + str(num_iter) + '_' + args.model + '_' + str(
        args.human_thres) + '_' + str(args.object_thres) + 'TIN&PP.pkl'
    merge_output = "1700000_TIN&partpair_average_0.20.pkl"

    # test detections result
    Test_RCNN      = pickle.load( open( cfg.DATA_DIR + '/' + 'my_Test.pkl', "rb" ) ,encoding='bytes')

    # pretrain model
    weight = cfg.ROOT_DIR + '/Weights/' + args.model + '/HOI_iter_' + str(num_iter) + '.ckpt'

    print ('Human thres = ' + str(args.human_thres) + ', Object thres = ' + str(args.object_thres) + ', iter = ' + str(args.iteration) + ', path = ' + weight )

    # init session
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    sess = tf.Session(config=tfconfig)

    net = ResNet50()
    net.create_architecture(False)

    # 恢复训练完网络的权重
    saver = tf.train.Saver()
    saver.restore(sess, weight)
    print('Pre-trained weights loaded.')

    # 开始测试
    test_net(sess, net, Test_RCNN, test_output, args.object_thres, args.human_thres)
    sess.close()

    # merge the best binary score and the best partpair score on TIN
    os.chdir(cfg.ROOT_DIR+'/-Results/')
    command1 = "python merge.py " + test_output + ' ' + merge_output
    os.system(command1)

    # thres_X and thres_Y indicate the NIS threshold to suppress the pair which might be no-interaction
    thres_x = 0.1
    thres_y = 0.9

    os.chdir(cfg.ROOT_DIR + '/HICO-DET_Benchmark/')
    merge_output= cfg.ROOT_DIR+'/-Results/'+merge_output
    # os.system: 将字符串转换为命令在服务器上运行
    # 命令为：Generate_HICO_detection_nis.py {cfg.ROOT_DIR}/-Results/1700000_TIN&partpair_aver \
    # age_0.20.pkl {cfg.ROOT_DIR}/-Results/TIN_HICO_testNIS_thres_x0.1_y0.9/ 0.9 0.1
    command2 = "python Generate_HICO_detection_nis.py " + merge_output + ' ' + cfg.ROOT_DIR + "/-Results/" + args.model + "_" + str(num_iter) + "_NIS_thres_x" + str(
        thres_x) + "_y" + str(thres_y) + "_average_0.20/ " + str(thres_y) + " " + str(thres_x)
    os.system(command2)