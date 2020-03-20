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

    test_output = "1700000_"+ args.model + "_0.8_0.3new_pair_selection_H2O2.pkl"
    merge_output = "1700000_"+ args.model + "_average_0.20.pkl"

    # merge the best binary score and the best partpair score on TIN prediction
    os.chdir(cfg.ROOT_DIR+'/-Results/')
    command1 = "python merge_binary_and_partpair.py " + test_output + ' ' + merge_output
    os.system(command1)

    # thres_X and thres_Y indicate the NIS threshold to suppress the pair which might be no-interaction
    thres_x = 0.1
    thres_y = 0.9

    os.chdir(cfg.ROOT_DIR + '/HICO-DET_Benchmark/')
    merge_output= cfg.ROOT_DIR+'/-Results/'+merge_output
    # os.system: 将字符串转换为命令在服务器上运行
    # 命令为：Generate_HICO_detection_nis.py {cfg.ROOT_DIR}/-Results/1700000_TIN_partpair_aver \
    # age_0.20.pkl {cfg.ROOT_DIR}/-Results/TIN_partpair_1700000_NIS_thres_x0.1_y0.9_average_0.20/ 0.9 0.1
    command2 = "python Generate_HICO_detection_nis.py " + merge_output + ' ' + cfg.ROOT_DIR + "/-Results/" + args.model + "_1700000_NIS_thres_x" + str(
        thres_x) + "_y" + str(thres_y) + "_average_0.20/ " + str(thres_y) + " " + str(thres_x)
    os.system(command2)