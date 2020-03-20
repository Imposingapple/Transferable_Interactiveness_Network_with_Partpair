# --------------------------------------------------------
# Tensorflow TIN
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.framework import ops

from ult.config import cfg
from ult.visualization import draw_bounding_boxes_HOI

import numpy as np
import ipdb
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# resnet中，对conv2d,fully_connected两函数默认属性的设置
def resnet_arg_scope(is_training=True,
                     weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):

    # batch_normalization的参数列表
    batch_norm_params = {
        'is_training': False,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': ops.GraphKeys.UPDATE_OPS
    }

    # slim模块设置了其中conv2d与fully_connected函数的默认属性
    # 其中conv2d与fully_connected函数都采用batch_normalization的方法对每个batch的输入经过该层的结果进行了正则化
    # 其中batch_normalization的参数用的是上面的字典中设置的值
    with arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY),
        weights_initializer = slim.variance_scaling_initializer(),
        biases_regularizer  = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), 
        biases_initializer  = tf.constant_initializer(0.0),
        trainable           = is_training,
        activation_fn       = tf.nn.relu,
        normalizer_fn       = slim.batch_norm,
        normalizer_params   = batch_norm_params):
        with arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc

class ResNet50(): # 64--128--256--512--512
    def __init__(self):
        self.visualize = {}
        self.intermediate = {}
        self.predictions = {}
        self.score_summaries = {}
        self.event_summaries = {}
        self.train_summaries = []
        self.losses = {}

        self.image       = tf.compat.v1.placeholder(tf.float32, shape=[1, None, None, 3], name = 'image')
        self.spatial     = tf.compat.v1.placeholder(tf.float32, shape=[None, 64, 64, 3], name = 'sp')
        self.H_boxes     = tf.compat.v1.placeholder(tf.float32, shape=[None, 5], name = 'H_boxes')
        self.O_boxes     = tf.compat.v1.placeholder(tf.float32, shape=[None, 5], name = 'O_boxes')
        self.S_boxes     = tf.placeholder(tf.float32, shape=[None, 5], name='S_boxes')
        self.P_boxes     = tf.placeholder(tf.float32, shape=[None, 10, 5], name='P_boxes')
        self.gt_class_HO = tf.compat.v1.placeholder(tf.float32, shape=[None, 600], name = 'gt_class_HO')
        self.gt_binary_label = tf.compat.v1.placeholder(tf.float32, shape=[None, 2], name = 'gt_binary_label')
        self.H_num       = tf.compat.v1.placeholder(tf.int32)
        self.HO_weight   = np.array([
                9.192927, 9.778443, 10.338059, 9.164914, 9.075144, 10.045923, 8.714437, 8.59822, 12.977117, 6.2745423, 
                11.227917, 6.765012, 9.436157, 9.56762, 11.0675745, 11.530198, 9.609821, 9.897503, 6.664475, 6.811699, 
                6.644726, 9.170454, 13.670264, 3.903943, 10.556748, 8.814335, 9.519224, 12.753973, 11.590822, 8.278912, 
                5.5245695, 9.7286825, 8.997436, 10.699849, 9.601237, 11.965516, 9.192927, 10.220277, 6.056692, 7.734048, 
                8.42324, 6.586457, 6.969533, 10.579222, 13.670264, 4.4531965, 9.326459, 9.288238, 8.071842, 10.431585, 
                12.417501, 11.530198, 11.227917, 4.0678477, 8.854023, 12.571651, 8.225684, 10.996116, 11.0675745, 10.100731, 
                7.0376034, 7.463688, 12.571651, 14.363411, 5.4902234, 11.0675745, 14.363411, 8.45805, 10.269067, 9.820116, 
                14.363411, 11.272368, 11.105314, 7.981595, 9.198626, 3.3284247, 14.363411, 12.977117, 9.300817, 10.032678, 
                12.571651, 10.114916, 10.471591, 13.264799, 14.363411, 8.01953, 10.412168, 9.644913, 9.981384, 7.2197933, 
                14.363411, 3.1178555, 11.031207, 8.934066, 7.546675, 6.386472, 12.060826, 8.862153, 9.799063, 12.753973, 
                12.753973, 10.412168, 10.8976755, 10.471591, 12.571651, 9.519224, 6.207762, 12.753973, 6.60636, 6.2896967, 
                4.5198326, 9.7887, 13.670264, 11.878505, 11.965516, 8.576513, 11.105314, 9.192927, 11.47304, 11.367679, 
                9.275815, 11.367679, 9.944571, 11.590822, 10.451388, 9.511381, 11.144535, 13.264799, 5.888291, 11.227917, 
                10.779892, 7.643191, 11.105314, 9.414651, 11.965516, 14.363411, 12.28397, 9.909063, 8.94731, 7.0330057, 
                8.129001, 7.2817025, 9.874775, 9.758241, 11.105314, 5.0690055, 7.4768796, 10.129305, 9.54313, 13.264799, 
                9.699972, 11.878505, 8.260853, 7.1437693, 6.9321113, 6.990665, 8.8104515, 11.655361, 13.264799, 4.515912, 
                9.897503, 11.418972, 8.113436, 8.795067, 10.236277, 12.753973, 14.363411, 9.352776, 12.417501, 0.6271591, 
                12.060826, 12.060826, 12.166186, 5.2946343, 11.318889, 9.8308115, 8.016022, 9.198626, 10.8976755, 13.670264, 
                11.105314, 14.363411, 9.653881, 9.503599, 12.753973, 5.80546, 9.653881, 9.592727, 12.977117, 13.670264, 
                7.995224, 8.639826, 12.28397, 6.586876, 10.929424, 13.264799, 8.94731, 6.1026597, 12.417501, 11.47304, 
                10.451388, 8.95624, 10.996116, 11.144535, 11.031207, 13.670264, 13.670264, 6.397866, 7.513285, 9.981384, 
                11.367679, 11.590822, 7.4348736, 4.415428, 12.166186, 8.573451, 12.977117, 9.609821, 8.601359, 9.055143, 
                11.965516, 11.105314, 13.264799, 5.8201604, 10.451388, 9.944571, 7.7855496, 14.363411, 8.5463, 13.670264, 
                7.9288645, 5.7561946, 9.075144, 9.0701065, 5.6871653, 11.318889, 10.252538, 9.758241, 9.407584, 13.670264, 
                8.570397, 9.326459, 7.488179, 11.798462, 9.897503, 6.7530537, 4.7828183, 9.519224, 7.6492405, 8.031909, 
                7.8180614, 4.451856, 10.045923, 10.83705, 13.264799, 13.670264, 4.5245686, 14.363411, 10.556748, 10.556748, 
                14.363411, 13.670264, 14.363411, 8.037262, 8.59197, 9.738439, 8.652985, 10.045923, 9.400566, 10.9622135, 
                11.965516, 10.032678, 5.9017305, 9.738439, 12.977117, 11.105314, 10.725825, 9.080208, 11.272368, 14.363411, 
                14.363411, 13.264799, 6.9279733, 9.153925, 8.075553, 9.126969, 14.363411, 8.903826, 9.488214, 5.4571533, 
                10.129305, 10.579222, 12.571651, 11.965516, 6.237189, 9.428937, 9.618479, 8.620408, 11.590822, 11.655361, 
                9.968962, 10.8080635, 10.431585, 14.363411, 3.796231, 12.060826, 10.302968, 9.551227, 8.75394, 10.579222, 
                9.944571, 14.363411, 6.272396, 10.625742, 9.690582, 13.670264, 11.798462, 13.670264, 11.724354, 9.993963, 
                8.230013, 9.100721, 10.374427, 7.865129, 6.514087, 14.363411, 11.031207, 11.655361, 12.166186, 7.419324, 
                9.421769, 9.653881, 10.996116, 12.571651, 13.670264, 5.912144, 9.7887, 8.585759, 8.272101, 11.530198, 8.886948, 
                5.9870906, 9.269661, 11.878505, 11.227917, 13.670264, 8.339964, 7.6763024, 10.471591, 10.451388, 13.670264, 
                11.185357, 10.032678, 9.313555, 12.571651, 3.993144, 9.379805, 9.609821, 14.363411, 9.709451, 8.965248, 
                10.451388, 7.0609145, 10.579222, 13.264799, 10.49221, 8.978916, 7.124196, 10.602211, 8.9743395, 7.77862, 
                8.073695, 9.644913, 9.339531, 8.272101, 4.794418, 9.016304, 8.012526, 10.674532, 14.363411, 7.995224, 
                12.753973, 5.5157638, 8.934066, 10.779892, 7.930471, 11.724354, 8.85808, 5.9025764, 14.363411, 12.753973, 
                12.417501, 8.59197, 10.513264, 10.338059, 14.363411, 7.7079706, 14.363411, 13.264799, 13.264799, 10.752493, 
                14.363411, 14.363411, 13.264799, 12.417501, 13.670264, 6.5661197, 12.977117, 11.798462, 9.968962, 12.753973, 
                11.47304, 11.227917, 7.6763024, 10.779892, 11.185357, 14.363411, 7.369478, 14.363411, 9.944571, 10.779892, 
                10.471591, 9.54313, 9.148476, 10.285873, 10.412168, 12.753973, 14.363411, 6.0308623, 13.670264, 10.725825, 
                12.977117, 11.272368, 7.663911, 9.137665, 10.236277, 13.264799, 6.715625, 10.9622135, 14.363411, 13.264799, 
                9.575919, 9.080208, 11.878505, 7.1863923, 9.366199, 8.854023, 9.874775, 8.2857685, 13.670264, 11.878505, 
                12.166186, 7.616999, 9.44343, 8.288065, 8.8104515, 8.347254, 7.4738197, 10.302968, 6.936267, 11.272368, 
                7.058223, 5.0138307, 12.753973, 10.173757, 9.863602, 11.318889, 9.54313, 10.996116, 12.753973, 7.8339925, 
                7.569945, 7.4427395, 5.560738, 12.753973, 10.725825, 10.252538, 9.307165, 8.491293, 7.9161053, 7.8849015, 
                7.782772, 6.3088884, 8.866243, 9.8308115, 14.363411, 10.8976755, 5.908519, 10.269067, 9.176025, 9.852551, 
                9.488214, 8.90809, 8.537411, 9.653881, 8.662968, 11.965516, 10.143904, 14.363411, 14.363411, 9.407584, 
                5.281472, 11.272368, 12.060826, 14.363411, 7.4135547, 8.920994, 9.618479, 8.891141, 14.363411, 12.060826, 
                11.965516, 10.9622135, 10.9622135, 14.363411, 5.658909, 8.934066, 12.571651, 8.614018, 11.655361, 13.264799, 
                10.996116, 13.670264, 8.965248, 9.326459, 11.144535, 14.363411, 6.0517673, 10.513264, 8.7430105, 10.338059, 
                13.264799, 6.878481, 9.065094, 8.87035, 14.363411, 9.92076, 6.5872955, 10.32036, 14.363411, 9.944571, 
                11.798462, 10.9622135, 11.031207, 7.652888, 4.334878, 13.670264, 13.670264, 14.363411, 10.725825, 12.417501, 
                14.363411, 13.264799, 11.655361, 10.338059, 13.264799, 12.753973, 8.206432, 8.916674, 8.59509, 14.363411, 
                7.376845, 11.798462, 11.530198, 11.318889, 11.185357, 5.0664344, 11.185357, 9.372978, 10.471591, 9.6629305, 
                11.367679, 8.73579, 9.080208, 11.724354, 5.04781, 7.3777695, 7.065643, 12.571651, 11.724354, 12.166186, 
                12.166186, 7.215852, 4.374113, 11.655361, 11.530198, 14.363411, 6.4993753, 11.031207, 8.344818, 10.513264, 
                10.032678, 14.363411, 14.363411, 4.5873594, 12.28397, 13.670264, 12.977117, 10.032678, 9.609821
            ], dtype = 'float32').reshape(1,600)
        self.binary_weight = np.array([1.6094379124341003, 0.22314355131420976], dtype = 'float32').reshape(1,2)
        self.num_classes = 600 # HOI
        self.num_binary  = 2 # existence (0 or 1) of HOI
        self.num_fc      = 1024
        self.transfer_mask_1 = np.ones((1, 600), dtype='float32')
        self.transfer_mask_2 = np.ones((1, 2), dtype='float32')
        self.scope       = 'resnet_v1_50'
        self.stride      = [16, ]
        self.lr          = tf.compat.v1.placeholder(tf.float32)

        self.blocks = [resnet_v1_block('block1', base_depth=64,  num_units=3, stride=2), # depth=256, depth_bottleneck=64
                       resnet_v1_block('block2', base_depth=128, num_units=4, stride=2), # depth=512, depth_bottleneck=128
                       resnet_v1_block('block3', base_depth=256, num_units=6, stride=1), # depth=1024, depth_bottleneck=256
                       # 到这之前的三个模块负责特征提取
                       resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
                       resnet_v1_block('block5', base_depth=512, num_units=3, stride=1),
                       resnet_v1_block('block6', base_depth=512, num_units=3, stride=1)] # 此block原本的resnet没有，新加上去的

    # 一个7*7，64个通道的卷积，一个max pool，将原图的通道数变成64，长宽变成原来的1/4
    def build_base(self):
        with tf.compat.v1.variable_scope(self.scope, self.scope):
            net = resnet_utils.conv2d_same(self.image, 64, 7, stride=2, scope='conv1')    # conv2d + subsample, 7*7的卷积核，步长为2，padding='SAME'，通道数变成64
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])     # 默认mode='CONSTANT，[0, 0], [1, 1], [1, 1], [0, 0]分别对输入的四个维度进行padding，即对图像长宽padding 1
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')

        return net

# Number of fixed blocks during training, by default ***the first of all 4 blocks*** is fixed (Resnet-50 block)
# Range: 0 (none) to 3 (all)
# __C.RESNET.FIXED_BLOCKS = 1

##########################################################################################################
#  for both TIN and Partpair
##########################################################################################################
    # 特征提取器，包括 base,block1,block2,block3，其中 base,block1中的参数的 trainable属性是 false
    # 输出形状为：(1, H, W, 1024)
    def image_to_head(self, is_training):
        # 模块参数固定，默认为base block和第一个Resnet block
        with slim.arg_scope(resnet_arg_scope(is_training=False)):
            net    = self.build_base()   # 经过 base 后形状为 (1, ?, ?, 64)
            net, _ = resnet_v1.resnet_v1(net,
                                         self.blocks[0:cfg.RESNET.FIXED_BLOCKS], 
                                         global_pool=False,
                                         include_root_block=False,
                                         scope=self.scope)      # 经过 block1 后形状为 (1, ?, ?, 256)

        # 模块参数由训练得到，把 block2,block3加到计算图中
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            head, _ = resnet_v1.resnet_v1(net,
                                          self.blocks[cfg.RESNET.FIXED_BLOCKS:-3],    ## 选取block2,block3，忽视block4,block5
                                          global_pool=False,
                                          include_root_block=False,
                                          scope=self.scope)
            # print("shape of head: ",head.shape)   # 经过 block3 后形状为 (1, ?, ?, 1024)
        return head

    # ROI pooling模块，将每个得到的物体框的 [xmin,ymin,xmax,ymax]，转换成7*7*1024的张量，其中1024是通过特征提取器后的通道数目
    # 输出形状为：（num_rois,7,7,1024) ,其中num_rois是输入的rois中包含的roi数量
    def crop_pool_layer(self, bottom, rois, name):
        with tf.compat.v1.variable_scope(name) as scope:
            # 先用tf.slice(rois, [0, 0], [-1, 1])，得到维度为（num_roi,1)的张量
            # 再使用tf.squeeze，得到该张图片的 num_roi
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])

            # 还原出图片的原始大小
            bottom_shape = tf.shape(bottom)
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self.stride[0])
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self.stride[0])

            # x1,y1,x2,y2都是长度为 num_roi 的一维张量，代表这些框在原图中的位置
            # 每个元素取值范围是（0,1），代表在图中的比例
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height

            # 在第一维度进行拼接，获得(num_roi,4)的张量
            # 二维张量中每个长度为4的一维张量，代表这些一个roi框在原图中的坐标
            # 通过stop_gradient函数，使得计算梯度时不考虑它们
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))

            if cfg.RESNET.MAX_POOL:
                pre_pool_size = cfg.POOLING_SIZE * 2
                crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids),
                                                 [pre_pool_size, pre_pool_size], name="crops")
                crops = slim.max_pool2d(crops, [2, 2], padding='SAME')
            # 我们使用到的是这里
            # 直接将1024个通道的特征图，每个通道上裁剪到的区域采样到7*7的区域
            # 最终得到的输出形状为（num_rois,7,7,1024)
            else:
                # crop_and_resize 函数输入的bbox形式是 [y1,x1,y2,x2]，其中每一个元素都是normalize完后在0-1间的数
                crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids),
                                                 [cfg.POOLING_SIZE, cfg.POOLING_SIZE], name="crops")
        return crops

##########################################################################################################
# for TIN
##########################################################################################################
    # Spatial Map空间提取模块，包括：conv-pool-conv-pool-flatten
    # 输出形状为：(num_pos_neg,5408)
    def sp_to_head(self):
        with tf.compat.v1.variable_scope(self.scope, self.scope):
            # (num_pos_neg,64,64,2)->(num_pos_neg,60,60,64)
            conv1_sp      = slim.conv2d(self.spatial[:,:,:,0:2], 64, [5, 5], padding='VALID', scope='conv1_sp')
            # (num_pos_neg,60,60,64)->(num_pos_neg,30,30,64)
            pool1_sp      = slim.max_pool2d(conv1_sp, [2, 2], scope='pool1_sp')
            # (num_pos_neg,30,30,64)->(num_pos_neg,26,26,32)
            conv2_sp      = slim.conv2d(pool1_sp,     32, [5, 5], padding='VALID', scope='conv2_sp')
            # (num_pos_neg,26,26,32)->(num_pos_neg,13,13,32)
            pool2_sp      = slim.max_pool2d(conv2_sp, [2, 2], scope='pool2_sp')
            # (num_pos_neg,13,13,32)->(num_pos_neg,5408)
            pool2_flat_sp = slim.flatten(pool2_sp)
        # print("shape of pool2_flat_sp: ",pool2_flat_sp.shape)   # 经过最后一个 max_pool2d 后形状为 (num_pos_neg,5408)
        return pool2_flat_sp

    # 在特征图上继续提取human和object的特征，human特征通过block4提取，object特征通过block5提取
    # 抽取出的human和object特征继续通过average pooling 对2048个通道，每个通道的所有7*7个点取平均
    # 输出形状为：(?, 2048)     '?' 对Human特征来说是num_pos_neg，对Object来说是num_pos
    def res5_TIN(self, pool5_H, pool5_O, is_training, name):
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            # block4负责提取 human 的特征
            fc7_H, _ = resnet_v1.resnet_v1(pool5_H, # H input, one block
                                           self.blocks[-3:-2],
                                           global_pool=False,
                                           include_root_block=False,
                                           reuse=False,
                                           scope=self.scope)
            fc7_H = tf.reduce_mean(fc7_H, axis=[1, 2])

            # block5负责提取 object 的特征
            fc7_O, _ = resnet_v1.resnet_v1(pool5_O, # O input, one block
                                       self.blocks[-2:-1],
                                       global_pool=False,
                                       include_root_block=False,
                                       reuse=False,
                                       scope=self.scope)
            fc7_O = tf.reduce_mean(fc7_O, axis=[1, 2])
        
        return fc7_H, fc7_O

    # 拥有5个输出：fc9_SH, fc9_SO, fc7_SHsp, fc7_SH, fc7_SO
    # fc9_SH:    Human  Stream（与attention机制获得的context information：fc7_SH结合之后），形状为 (num_pos_neg,1024)
    # fc9_SO:    Object Stream（与attention机制获得的context information：fc7_SO结合之后），形状为 (num_pos,1024)
    # fc7_SHsp:  Human+Spatial Stream，形状为(num_pos_neg,1024)
    # fc7_SH:    Human 的contextual information，形状为(num_pos_neg,1024)
    # fc7_SO:    Object的contextual information，形状为(num_pos,1024)
    def head_to_tail(self, fc7_H, fc7_O, pool5_SH, pool5_SO, sp, is_training, name):
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            # pool5_SH 猜测形状为：（num_pos_neg, H, W, 1024)
            # fc7_H 猜测形状为：（num_pos_neg, 2048)
            fc7_SH = tf.reduce_mean(pool5_SH, axis=[1, 2])    # 形状为： (?, 1024)
            fc7_SO = tf.reduce_mean(pool5_SO, axis=[1, 2])

            # 得到 Human Stream，形状为 (num_pos_neg,3072)
            Concat_SH     = tf.concat([fc7_H, fc7_SH], 1)     # 形状为： (?, 3072), 猜测是(num_pos_neg,3072)
            fc8_SH        = slim.fully_connected(Concat_SH, self.num_fc, scope='fc8_SH') #fc size = 1024
            fc8_SH        = slim.dropout(fc8_SH, keep_prob=0.5, is_training=is_training, scope='dropout8_SH')
            fc9_SH        = slim.fully_connected(fc8_SH, self.num_fc, scope='fc9_SH')
            fc9_SH        = slim.dropout(fc9_SH, keep_prob=0.5, is_training=is_training, scope='dropout9_SH')  # 形状为： (?, 1024), 猜测是(num_pos_neg,1024)

            # 得到 Object Stream，形状为 (num_pos,3072)
            Concat_SO     = tf.concat([fc7_O, fc7_SO], 1)
            fc8_SO        = slim.fully_connected(Concat_SO, self.num_fc, scope='fc8_SO')
            fc8_SO        = slim.dropout(fc8_SO, keep_prob=0.5, is_training=is_training, scope='dropout8_SO')
            fc9_SO        = slim.fully_connected(fc8_SO, self.num_fc, scope='fc9_SO')
            fc9_SO        = slim.dropout(fc9_SO, keep_prob=0.5, is_training=is_training, scope='dropout9_SO')

            # 输入是提取出人物框的特征的2048维张量和 H-O Spatial Map的5408维向量
            # 得到 Human+Spatial Stream，形状为 (num_pos_neg,7456)
            Concat_SHsp   = tf.concat([fc7_H, sp], 1)   # 形状为： (?,7456), 猜测是(num_pos_neg,7456)
            Concat_SHsp   = slim.fully_connected(Concat_SHsp, self.num_fc, scope='Concat_SHsp')
            Concat_SHsp   = slim.dropout(Concat_SHsp, keep_prob=0.5, is_training=is_training, scope='dropout6_SHsp')
            fc7_SHsp      = slim.fully_connected(Concat_SHsp, self.num_fc, scope='fc7_SHsp')
            fc7_SHsp      = slim.dropout(fc7_SHsp,  keep_prob=0.5, is_training=is_training, scope='dropout7_SHsp')

        return fc9_SH, fc9_SO, fc7_SHsp, fc7_SH, fc7_SO

    # binary discriminator for 0/1 classification of interaction, fc7_H, fc7_SH, fc7_O, fc7_SO, sp
    # 输出形状为：(num_pos_neg,1024)
    def binary_discriminator(self, fc7_H, fc7_O, fc7_SH, fc7_SO, sp, is_training, name):
        with tf.compat.v1.variable_scope(name) as scope:
            # 每一幅图片的num_pos_neg个H-O pair，每个pair的pose信息被拉成2704维向量
            conv1_pose_map      = slim.conv2d(self.spatial[:,:,:,2:], 32, [5, 5], padding='VALID', scope='conv1_pose_map')  # self.spatial[:,:,:,2:]的形状为：(num_pos_neg,64,64,1)
            pool1_pose_map      = slim.max_pool2d(conv1_pose_map, [2, 2], scope='pool1_pose_map')
            conv2_pose_map      = slim.conv2d(pool1_pose_map,     16, [5, 5], padding='VALID', scope='conv2_pose_map')
            pool2_pose_map      = slim.max_pool2d(conv2_pose_map, [2, 2], scope='pool2_pose_map')   # 形状为：(num_pos_neg,13,13,16)
            pool2_flat_pose_map = slim.flatten(pool2_pose_map)   # 形状为：(num_pos_neg,2704)

            # fc7_H + fc7_SH + sp + pose ---fc1024---fc8_binary_1
            # fc_binary_1 融合 Human appearance feature, Human context feature, H-O对的空间位置信息, 人物姿态信息
            # fc8_binary_1将所得fc_binary_1进行全连接，得到每个H-O pair中 Human Stream 的最终输出
            fc_binary_1    = tf.concat([fc7_H, fc7_SH], 1) # [pos + neg, 3072]
            fc_binary_1    = tf.concat([fc_binary_1, sp, pool2_flat_pose_map], 1)   # 形状为：(num_pos_neg, 11184)
            fc8_binary_1     = slim.fully_connected(fc_binary_1, 1024, scope = 'fc8_binary_1') 
            fc8_binary_1     = slim.dropout(fc8_binary_1, keep_prob = cfg.TRAIN_DROP_OUT_BINARY, is_training = is_training, scope = 'dropout8_binary_1') # [pos + neg,1024]

            # fc7_O + fc7_SO---fc1024---fc8_binary_2
            fc_binary_2    = tf.concat([fc7_O, fc7_SO], 1) # [pos, 3072]
            fc8_binary_2     = slim.fully_connected(fc_binary_2, 1024, scope = 'fc8_binary_2')
            fc8_binary_2     = slim.dropout(fc8_binary_2, keep_prob = cfg.TRAIN_DROP_OUT_BINARY, is_training = is_training, scope = 'dropout8_binary_2') # [pos,1024]
            fc8_binary_2   = tf.concat([fc8_binary_2, fc8_binary_1[self.H_num:,:]], 0)   # 补充 batch 数量至num_pos_neg

            # fc8_binary_1 + fc8_binary_2---fc1024---fc9_binary
            fc8_binary     = tf.concat([fc8_binary_1, fc8_binary_2], 1)
            fc9_binary     = slim.fully_connected(fc8_binary, 1024, scope = 'fc9_binary')
            fc9_binary     = slim.dropout(fc9_binary, keep_prob = cfg.TRAIN_DROP_OUT_BINARY, is_training = is_training, scope = 'dropout9_binary')
        return fc9_binary

    # P网络头，biary classification，接受concat后的(num_pos_neg,1024)维张量
    # Interactive score(经过sigmoid前)写入：self.predictions["cls_score_binary"]
    # Interactive probability(经过sigmoid后)写入：self.predictions["cls_prob_binary"]
    # 输出cls_prob_binary，即网络关于是否存在Interactiveness的预测，形状为: (?,2)
    def binary_classification(self, fc9_binary, is_training, initializer, name):
        with tf.compat.v1.variable_scope(name) as scope:
            cls_score_binary = slim.fully_connected(fc9_binary, self.num_binary, 
                                               weights_initializer=initializer,
                                               trainable=is_training,
                                               activation_fn=None, scope='cls_score_binary')
            cls_prob_binary  = tf.nn.sigmoid(cls_score_binary, name='cls_prob_binary')   # 形状为：(?,2)

            self.predictions["cls_score_binary"]  = cls_score_binary
            self.predictions["cls_prob_binary"]  = cls_prob_binary
        return cls_prob_binary

    def attention_pool_layer_H(self, bottom, fc7_H, is_training, name):
        with tf.compat.v1.variable_scope(name) as scope:
            fc1         = slim.fully_connected(fc7_H, 512, scope='fc1_b')   # 形状为（?, 512)
            fc1         = slim.dropout(fc1, keep_prob=0.8, is_training=is_training, scope='dropout1_b')  # 形状为（?,512),猜测为(num_pos_neg,512)
            fc1         = tf.reshape(fc1, [tf.shape(fc1)[0], 1, 1, tf.shape(fc1)[1]])  # 形状为（?, 1, 1, ?),猜测为(num_pos_neg,1,1,512)
            # tf.multiply()：两矩阵对应元素相乘，这里长、宽两个维度用到了广播机制
            # bottom为经过bottleneck后的，猜测形状为(1,H,W,512)的特征图
            # 我们对经过multiply后的新的特征图，每个位置的512个通道求平均
            att         = tf.reduce_mean(tf.multiply(bottom, fc1), 3, keep_dims=True)  # 形状为（?,?,?,1),猜测为（num_pos_neg,H,W,1)
        return att

    def attention_norm_H(self, att, name):
        with tf.compat.v1.variable_scope(name) as scope:
            att         = tf.transpose(att, [0, 3, 1, 2])
            att_shape   = tf.shape(att)
            att         = tf.reshape(att, [att_shape[0], att_shape[1], -1])   # 猜测是（num_pos_neg, 1, H*W)
            att         = tf.nn.softmax(att)
            att         = tf.reshape(att, att_shape)
            att         = tf.transpose(att, [0, 2, 3, 1])
        return att

    def attention_pool_layer_O(self, bottom, fc7_O, is_training, name):
        with tf.compat.v1.variable_scope(name) as scope:

            fc1         = slim.fully_connected(fc7_O, 512, scope='fc1_b')
            fc1         = slim.dropout(fc1, keep_prob=0.8, is_training=is_training, scope='dropout1_b')
            fc1         = tf.reshape(fc1, [tf.shape(fc1)[0], 1, 1, tf.shape(fc1)[1]])
            att         = tf.reduce_mean(tf.multiply(bottom, fc1), 3, keep_dims=True)
        return att

    def attention_norm_O(self, att, name):
        with tf.compat.v1.variable_scope(name) as scope:

            att         = tf.transpose(att, [0, 3, 1, 2])
            att_shape   = tf.shape(att)
            att         = tf.reshape(att, [att_shape[0], att_shape[1], -1])
            att         = tf.nn.softmax(att) ###
            att         = tf.reshape(att, att_shape)
            att         = tf.transpose(att, [0, 2, 3, 1])
        return att

    def region_classification_TIN(self, fc9_SH, fc9_SO, fc7_SHsp, is_training, initializer, name):
        with tf.compat.v1.variable_scope(name) as scope:
            cls_score_H = slim.fully_connected(fc9_SH, self.num_classes,
                                               weights_initializer=initializer,
                                               trainable=is_training,
                                               activation_fn=None, scope='cls_score_H')
            cls_prob_H  = tf.nn.sigmoid(cls_score_H, name='cls_prob_H') 
            tf.reshape(cls_prob_H, [1, self.num_classes]) 

            cls_score_O  = slim.fully_connected(fc9_SO, self.num_classes,
                                               weights_initializer=initializer,
                                               trainable=is_training,
                                               activation_fn=None, scope='cls_score_O')
            cls_prob_O  = tf.nn.sigmoid(cls_score_O, name='cls_prob_O') 
            tf.reshape(cls_prob_O, [1, self.num_classes]) 

            cls_score_sp = slim.fully_connected(fc7_SHsp, self.num_classes, 
                                               weights_initializer=initializer,
                                               trainable=is_training,
                                               activation_fn=None, scope='cls_score_sp')
            cls_prob_sp  = tf.nn.sigmoid(cls_score_sp, name='cls_prob_sp') 
            tf.reshape(cls_prob_sp, [1, self.num_classes])

            self.predictions["cls_score_H"]  = cls_score_H
            self.predictions["cls_prob_H"]   = cls_prob_H
            self.predictions["cls_score_O"]  = cls_score_O
            self.predictions["cls_prob_O"]   = cls_prob_O
            self.predictions["cls_score_sp"] = cls_score_sp
            self.predictions["cls_prob_sp"]  = cls_prob_sp

            self.predictions["cls_prob_TIN"]  = cls_prob_sp * (cls_prob_O + cls_prob_H) # late fusion of predictions

        return cls_prob_H, cls_prob_O, cls_prob_sp

    # 只有一层卷积，改变通道数
    def bottleneck(self, bottom, is_training, name, reuse=False):
        with tf.compat.v1.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            head_bottleneck = slim.conv2d(bottom, 1024, [1, 1], scope=name) # 1x1, 1024, fc
        return head_bottleneck

##########################################################################################################
# for Partpair
##########################################################################################################
    def res5_partpair(self, pool5_H, is_training, name):
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)): # updating

            pool5_H, _ = resnet_v1.resnet_v1(pool5_H, # H input, one block
                                           self.blocks[-1:], #fourth block
                                           global_pool=False,
                                           include_root_block=False,
                                           reuse=False,
                                           scope=self.scope)

        return pool5_H
    # 接受两个[batch_size,5]的框，两个框分别为 Human box和 Object box
    # 返回一个[batch_size,5]的 union框
    # 每个框：[0,x1,y1,x2,y2]
    def get_union_box(self, box_1, box_2):
        # with tf.variable_scope(name) as scope:
        # box: [batch, 5], left-up-(0,0) ,每个box为：[0, x1, y1, x2, y2]
        # union box = min_x1, min_y1, max_x2, max_y2
        H_boxes_x1 = box_1[:, 1] #[pos+neg]
        H_boxes_y1 = box_1[:, 2]
        H_boxes_x2 = box_1[:, 3]
        H_boxes_y2 = box_1[:, 4]
        H_boxes_x1=tf.reshape(H_boxes_x1, [tf.shape(H_boxes_x1)[0], 1])#[pos+neg, 1]
        H_boxes_y1=tf.reshape(H_boxes_y1, [tf.shape(H_boxes_y1)[0], 1])
        H_boxes_x2=tf.reshape(H_boxes_x2, [tf.shape(H_boxes_x2)[0], 1])
        H_boxes_y2=tf.reshape(H_boxes_y2, [tf.shape(H_boxes_y2)[0], 1])

        O_boxes_x1 = box_2[:, 1]
        O_boxes_y1 = box_2[:, 2]
        O_boxes_x2 = box_2[:, 3]
        O_boxes_y2 = box_2[:, 4]
        O_boxes_x1=tf.reshape(O_boxes_x1, [tf.shape(O_boxes_x1)[0], 1])
        O_boxes_y1=tf.reshape(O_boxes_y1, [tf.shape(O_boxes_y1)[0], 1])
        O_boxes_x2=tf.reshape(O_boxes_x2, [tf.shape(O_boxes_x2)[0], 1])
        O_boxes_y2=tf.reshape(O_boxes_y2, [tf.shape(O_boxes_y2)[0], 1])

        union_x1 = tf.concat([H_boxes_x1, O_boxes_x1], 1) #[pos+neg, 2]
        union_y1 = tf.concat([H_boxes_y1, O_boxes_y1], 1)
        union_x2 = tf.concat([H_boxes_x2, O_boxes_x2], 1)
        union_y2 = tf.concat([H_boxes_y2, O_boxes_y2], 1)
        union_boxes_x1 = tf.reduce_min(union_x1, 1)  #[pos+neg]
        union_boxes_y1 = tf.reduce_min(union_y1, 1)
        union_boxes_x2 = tf.reduce_max(union_x2, 1)
        union_boxes_y2 = tf.reduce_max(union_y2, 1)

        union_boxes_zero = self.H_boxes[:, 0] #[pos+neg]
        union_boxes_zero = tf.reshape(union_boxes_zero, [tf.shape(union_boxes_zero)[0], 1])#[pos+neg, 1]
        union_boxes_x1=tf.reshape(union_boxes_x1, [tf.shape(union_boxes_x1)[0], 1])
        union_boxes_y1=tf.reshape(union_boxes_y1, [tf.shape(union_boxes_y1)[0], 1])
        union_boxes_x2=tf.reshape(union_boxes_x2, [tf.shape(union_boxes_x2)[0], 1])
        union_boxes_y2=tf.reshape(union_boxes_y2, [tf.shape(union_boxes_y2)[0], 1])

        union_boxes = tf.stop_gradient(tf.concat([union_boxes_zero, union_boxes_x1, union_boxes_y1, union_boxes_x2, union_boxes_y2], 1)) #[pos+neg, 5]

        return union_boxes

    # 接受一个形状为 (1, H, W, 256) 的 head_part_pair作为 feature map, 还接受 [pos+neg,10,5]的人物的身体部位的位置信息
    # 对我们设定的30种 pair组合，每一种获取这个 pair两个部位位置的 union box，然后在这个 union box里通过采样获得 7*7的特征
    # 一个 Human的每个 pair，返回一个 7*7*256的特征，我们将30个 pair在通道层面叠加，获得 7*7*7680 的特征
    # 返回形状为： [pos+neg,7,7,7680]
    def ROI_for_part_pair(self, head_part_pair, P_boxes, name):
        with tf.variable_scope(name) as scope:
            # 30 pairs: 1-7, 1-10, 4-7, 4-10, 1-2, 1-3, 4-2, 4-3, 1-8, 1-9, 4-8, 4-9, 1-5, 4-5, 2-7, 3-7, 2-10, 3-10, 2-8, 3-8, 2-9, 3-9, 2-5, 3-5, 5-7, 5-10, 6-7, 6-10, 6-8, 6-9
            # in Px   : 3-9, 3-6,  0-9, 0-6,  3-2, 3-1, 0-2, 0-1, 3-8, 3-7, 0-8, 0-7, 3-4, 0-4, 2-9, 1-9, 2-6,  1-6,  2-8, 1-8, 2-7, 1-7, 2-4, 1-4, 4-9, 4-6,  5-9, 5-6,  5-8, 5-7
            # P_boxes-[pos_neg, 10, 5], P_boxes[:, 0, :]-[pos_neg, 5]
            pair_box_01 = self.get_union_box(P_boxes[:, 3, :], P_boxes[:, 9, :])
            pair_box_02 = self.get_union_box(P_boxes[:, 3, :], P_boxes[:, 6, :])
            pair_box_03 = self.get_union_box(P_boxes[:, 0, :], P_boxes[:, 9, :])
            pair_box_04 = self.get_union_box(P_boxes[:, 0, :], P_boxes[:, 6, :])
            pair_box_05 = self.get_union_box(P_boxes[:, 3, :], P_boxes[:, 2, :])
            pair_box_06 = self.get_union_box(P_boxes[:, 3, :], P_boxes[:, 1, :])
            pair_box_07 = self.get_union_box(P_boxes[:, 0, :], P_boxes[:, 2, :])
            pair_box_08 = self.get_union_box(P_boxes[:, 0, :], P_boxes[:, 1, :])
            pair_box_09 = self.get_union_box(P_boxes[:, 3, :], P_boxes[:, 8, :])
            pair_box_10 = self.get_union_box(P_boxes[:, 3, :], P_boxes[:, 7, :])
            pair_box_11 = self.get_union_box(P_boxes[:, 0, :], P_boxes[:, 8, :])
            pair_box_12 = self.get_union_box(P_boxes[:, 0, :], P_boxes[:, 7, :])
            pair_box_13 = self.get_union_box(P_boxes[:, 3, :], P_boxes[:, 4, :])
            pair_box_14 = self.get_union_box(P_boxes[:, 0, :], P_boxes[:, 4, :])
            pair_box_15 = self.get_union_box(P_boxes[:, 2, :], P_boxes[:, 9, :])
            pair_box_16 = self.get_union_box(P_boxes[:, 1, :], P_boxes[:, 9, :])
            pair_box_17 = self.get_union_box(P_boxes[:, 2, :], P_boxes[:, 6, :])
            pair_box_18 = self.get_union_box(P_boxes[:, 1, :], P_boxes[:, 6, :])
            pair_box_19 = self.get_union_box(P_boxes[:, 2, :], P_boxes[:, 8, :])
            pair_box_20 = self.get_union_box(P_boxes[:, 1, :], P_boxes[:, 8, :])
            pair_box_21 = self.get_union_box(P_boxes[:, 2, :], P_boxes[:, 7, :])
            pair_box_22 = self.get_union_box(P_boxes[:, 1, :], P_boxes[:, 7, :])
            pair_box_23 = self.get_union_box(P_boxes[:, 2, :], P_boxes[:, 4, :])
            pair_box_24 = self.get_union_box(P_boxes[:, 1, :], P_boxes[:, 4, :])
            pair_box_25 = self.get_union_box(P_boxes[:, 4, :], P_boxes[:, 9, :])
            pair_box_26 = self.get_union_box(P_boxes[:, 4, :], P_boxes[:, 6, :])
            pair_box_27 = self.get_union_box(P_boxes[:, 5, :], P_boxes[:, 9, :])
            pair_box_28 = self.get_union_box(P_boxes[:, 5, :], P_boxes[:, 6, :])
            pair_box_29 = self.get_union_box(P_boxes[:, 5, :], P_boxes[:, 8, :])
            pair_box_30 = self.get_union_box(P_boxes[:, 5, :], P_boxes[:, 7, :])

            pool5_pair_box_01 = self.crop_pool_layer(head_part_pair, pair_box_01, 'crop_pair_01')
            pool5_pair_box_02 = self.crop_pool_layer(head_part_pair, pair_box_02, 'crop_pair_02')
            pool5_pair_box_03 = self.crop_pool_layer(head_part_pair, pair_box_03, 'crop_pair_03')
            pool5_pair_box_04 = self.crop_pool_layer(head_part_pair, pair_box_04, 'crop_pair_04')
            pool5_pair_box_05 = self.crop_pool_layer(head_part_pair, pair_box_05, 'crop_pair_05')
            pool5_pair_box_06 = self.crop_pool_layer(head_part_pair, pair_box_06, 'crop_pair_06')
            pool5_pair_box_07 = self.crop_pool_layer(head_part_pair, pair_box_07, 'crop_pair_07')
            pool5_pair_box_08 = self.crop_pool_layer(head_part_pair, pair_box_08, 'crop_pair_08')
            pool5_pair_box_09 = self.crop_pool_layer(head_part_pair, pair_box_09, 'crop_pair_09')
            pool5_pair_box_10 = self.crop_pool_layer(head_part_pair, pair_box_10, 'crop_pair_10')
            pool5_pair_box_11 = self.crop_pool_layer(head_part_pair, pair_box_11, 'crop_pair_11')
            pool5_pair_box_12 = self.crop_pool_layer(head_part_pair, pair_box_12, 'crop_pair_12')
            pool5_pair_box_13 = self.crop_pool_layer(head_part_pair, pair_box_13, 'crop_pair_13')
            pool5_pair_box_14 = self.crop_pool_layer(head_part_pair, pair_box_14, 'crop_pair_14')
            pool5_pair_box_15 = self.crop_pool_layer(head_part_pair, pair_box_15, 'crop_pair_15')
            pool5_pair_box_16 = self.crop_pool_layer(head_part_pair, pair_box_16, 'crop_pair_16')
            pool5_pair_box_17 = self.crop_pool_layer(head_part_pair, pair_box_17, 'crop_pair_17')
            pool5_pair_box_18 = self.crop_pool_layer(head_part_pair, pair_box_18, 'crop_pair_18')
            pool5_pair_box_19 = self.crop_pool_layer(head_part_pair, pair_box_19, 'crop_pair_19')
            pool5_pair_box_20 = self.crop_pool_layer(head_part_pair, pair_box_20, 'crop_pair_20')
            pool5_pair_box_21 = self.crop_pool_layer(head_part_pair, pair_box_21, 'crop_pair_21')
            pool5_pair_box_22 = self.crop_pool_layer(head_part_pair, pair_box_22, 'crop_pair_22')
            pool5_pair_box_23 = self.crop_pool_layer(head_part_pair, pair_box_23, 'crop_pair_23')
            pool5_pair_box_24 = self.crop_pool_layer(head_part_pair, pair_box_24, 'crop_pair_24')
            pool5_pair_box_25 = self.crop_pool_layer(head_part_pair, pair_box_25, 'crop_pair_25')
            pool5_pair_box_26 = self.crop_pool_layer(head_part_pair, pair_box_26, 'crop_pair_26')
            pool5_pair_box_27 = self.crop_pool_layer(head_part_pair, pair_box_27, 'crop_pair_27')
            pool5_pair_box_28 = self.crop_pool_layer(head_part_pair, pair_box_28, 'crop_pair_28')
            pool5_pair_box_29 = self.crop_pool_layer(head_part_pair, pair_box_29, 'crop_pair_29')
            pool5_pair_box_30 = self.crop_pool_layer(head_part_pair, pair_box_30, 'crop_pair_30') # [pos_neg, 7, 7, 256]

            pool5_part_pairs = tf.concat([pool5_pair_box_01, pool5_pair_box_02, pool5_pair_box_03, pool5_pair_box_04, pool5_pair_box_05, pool5_pair_box_06, pool5_pair_box_07,
                                          pool5_pair_box_08, pool5_pair_box_09, pool5_pair_box_10, pool5_pair_box_11, pool5_pair_box_12, pool5_pair_box_13, pool5_pair_box_14,
                                          pool5_pair_box_15, pool5_pair_box_16, pool5_pair_box_17, pool5_pair_box_18, pool5_pair_box_19, pool5_pair_box_20, pool5_pair_box_21,
                                          pool5_pair_box_22, pool5_pair_box_23, pool5_pair_box_24, pool5_pair_box_25, pool5_pair_box_26, pool5_pair_box_27, pool5_pair_box_28,
                                          pool5_pair_box_29, pool5_pair_box_30], axis=3)

        return pool5_part_pairs

    # 为30个body pair每个得到的256个通道的feature map加上attention score(每个body pair对应一个7*7的attention score)
    # 输出形状为： [pos+neg,7,7,7680]
    def part_pairs_attention(self, pool5_part_pairs, is_training, initializer, name):
        with tf.variable_scope(name) as scope:
            fc8_part_pair_att      = slim.fully_connected(pool5_part_pairs, 4096, weights_initializer=initializer, trainable=is_training)
            fc8_part_pair_att       = slim.dropout(fc8_part_pair_att, keep_prob=0.5, is_training=is_training)
            fc9_part_pair_att       = slim.fully_connected(fc8_part_pair_att, 4096, weights_initializer=initializer, trainable=is_training) # [pos_neg, 7, 7, 4096]
            fc9_part_pair_att       = slim.dropout(fc9_part_pair_att, keep_prob=0.5, is_training=is_training)  # [pos+neg,7,7,4096]
            # print("shape of fc9_part_pair_att: ", fc9_part_pair_att)

            cls_score_part_pair_att = slim.fully_connected(fc9_part_pair_att, 30,
                                               weights_initializer=initializer,
                                               trainable=is_training,
                                               activation_fn=None)
            cls_prob_part_pair_att   = tf.nn.sigmoid(cls_score_part_pair_att) # [pos_neg, 7, 7, 30]
            # slice and attention
            # 最后的[1,1,1,256]表示在前三个维度维持原样，最后一个维度重复256次
            # pair_att_xx的形状为 [pos+neg,7,7,256]，代表一种肢体组合的att分数
            pair_att_01 = tf.tile(tf.reshape(cls_prob_part_pair_att[:, :, :, 0], [tf.shape(cls_prob_part_pair_att)[0], 7, 7, 1]), [1, 1, 1, 256])
            pair_att_02 = tf.tile(tf.reshape(cls_prob_part_pair_att[:, :, :, 1], [tf.shape(cls_prob_part_pair_att)[0], 7, 7, 1]), [1, 1, 1, 256])
            pair_att_03 = tf.tile(tf.reshape(cls_prob_part_pair_att[:, :, :, 2], [tf.shape(cls_prob_part_pair_att)[0], 7, 7, 1]), [1, 1, 1, 256])
            pair_att_04 = tf.tile(tf.reshape(cls_prob_part_pair_att[:, :, :, 3], [tf.shape(cls_prob_part_pair_att)[0], 7, 7, 1]), [1, 1, 1, 256])
            pair_att_05 = tf.tile(tf.reshape(cls_prob_part_pair_att[:, :, :, 4], [tf.shape(cls_prob_part_pair_att)[0], 7, 7, 1]), [1, 1, 1, 256])
            pair_att_06 = tf.tile(tf.reshape(cls_prob_part_pair_att[:, :, :, 5], [tf.shape(cls_prob_part_pair_att)[0], 7, 7, 1]), [1, 1, 1, 256])
            pair_att_07 = tf.tile(tf.reshape(cls_prob_part_pair_att[:, :, :, 6], [tf.shape(cls_prob_part_pair_att)[0], 7, 7, 1]), [1, 1, 1, 256])
            pair_att_08 = tf.tile(tf.reshape(cls_prob_part_pair_att[:, :, :, 7], [tf.shape(cls_prob_part_pair_att)[0], 7, 7, 1]), [1, 1, 1, 256])
            pair_att_09 = tf.tile(tf.reshape(cls_prob_part_pair_att[:, :, :, 8], [tf.shape(cls_prob_part_pair_att)[0], 7, 7, 1]), [1, 1, 1, 256])
            pair_att_10 = tf.tile(tf.reshape(cls_prob_part_pair_att[:, :, :, 9], [tf.shape(cls_prob_part_pair_att)[0], 7, 7, 1]), [1, 1, 1, 256])
            pair_att_11 = tf.tile(tf.reshape(cls_prob_part_pair_att[:, :, :, 10], [tf.shape(cls_prob_part_pair_att)[0], 7, 7, 1]), [1, 1, 1, 256])
            pair_att_12 = tf.tile(tf.reshape(cls_prob_part_pair_att[:, :, :, 11], [tf.shape(cls_prob_part_pair_att)[0], 7, 7, 1]), [1, 1, 1, 256])
            pair_att_13 = tf.tile(tf.reshape(cls_prob_part_pair_att[:, :, :, 12], [tf.shape(cls_prob_part_pair_att)[0], 7, 7, 1]), [1, 1, 1, 256])
            pair_att_14 = tf.tile(tf.reshape(cls_prob_part_pair_att[:, :, :, 13], [tf.shape(cls_prob_part_pair_att)[0], 7, 7, 1]), [1, 1, 1, 256])
            pair_att_15 = tf.tile(tf.reshape(cls_prob_part_pair_att[:, :, :, 14], [tf.shape(cls_prob_part_pair_att)[0], 7, 7, 1]), [1, 1, 1, 256])
            pair_att_16 = tf.tile(tf.reshape(cls_prob_part_pair_att[:, :, :, 15], [tf.shape(cls_prob_part_pair_att)[0], 7, 7, 1]), [1, 1, 1, 256])
            pair_att_17 = tf.tile(tf.reshape(cls_prob_part_pair_att[:, :, :, 16], [tf.shape(cls_prob_part_pair_att)[0], 7, 7, 1]), [1, 1, 1, 256])
            pair_att_18 = tf.tile(tf.reshape(cls_prob_part_pair_att[:, :, :, 17], [tf.shape(cls_prob_part_pair_att)[0], 7, 7, 1]), [1, 1, 1, 256])
            pair_att_19 = tf.tile(tf.reshape(cls_prob_part_pair_att[:, :, :, 18], [tf.shape(cls_prob_part_pair_att)[0], 7, 7, 1]), [1, 1, 1, 256])
            pair_att_20 = tf.tile(tf.reshape(cls_prob_part_pair_att[:, :, :, 19], [tf.shape(cls_prob_part_pair_att)[0], 7, 7, 1]), [1, 1, 1, 256])
            pair_att_21 = tf.tile(tf.reshape(cls_prob_part_pair_att[:, :, :, 20], [tf.shape(cls_prob_part_pair_att)[0], 7, 7, 1]), [1, 1, 1, 256])
            pair_att_22 = tf.tile(tf.reshape(cls_prob_part_pair_att[:, :, :, 21], [tf.shape(cls_prob_part_pair_att)[0], 7, 7, 1]), [1, 1, 1, 256])
            pair_att_23 = tf.tile(tf.reshape(cls_prob_part_pair_att[:, :, :, 22], [tf.shape(cls_prob_part_pair_att)[0], 7, 7, 1]), [1, 1, 1, 256])
            pair_att_24 = tf.tile(tf.reshape(cls_prob_part_pair_att[:, :, :, 23], [tf.shape(cls_prob_part_pair_att)[0], 7, 7, 1]), [1, 1, 1, 256])
            pair_att_25 = tf.tile(tf.reshape(cls_prob_part_pair_att[:, :, :, 24], [tf.shape(cls_prob_part_pair_att)[0], 7, 7, 1]), [1, 1, 1, 256])
            pair_att_26 = tf.tile(tf.reshape(cls_prob_part_pair_att[:, :, :, 25], [tf.shape(cls_prob_part_pair_att)[0], 7, 7, 1]), [1, 1, 1, 256])
            pair_att_27 = tf.tile(tf.reshape(cls_prob_part_pair_att[:, :, :, 26], [tf.shape(cls_prob_part_pair_att)[0], 7, 7, 1]), [1, 1, 1, 256])
            pair_att_28 = tf.tile(tf.reshape(cls_prob_part_pair_att[:, :, :, 27], [tf.shape(cls_prob_part_pair_att)[0], 7, 7, 1]), [1, 1, 1, 256])
            pair_att_29 = tf.tile(tf.reshape(cls_prob_part_pair_att[:, :, :, 28], [tf.shape(cls_prob_part_pair_att)[0], 7, 7, 1]), [1, 1, 1, 256])
            pair_att_30 = tf.tile(tf.reshape(cls_prob_part_pair_att[:, :, :, 29], [tf.shape(cls_prob_part_pair_att)[0], 7, 7, 1]), [1, 1, 1, 256])
            pair_att = tf.concat([pair_att_01, pair_att_02, pair_att_03, pair_att_04, pair_att_05, pair_att_06, pair_att_07, pair_att_08, pair_att_09, pair_att_10,
                                  pair_att_11, pair_att_12, pair_att_13, pair_att_14, pair_att_15, pair_att_16, pair_att_17, pair_att_18, pair_att_19, pair_att_20,
                                  pair_att_21, pair_att_22, pair_att_23, pair_att_24, pair_att_25, pair_att_26, pair_att_27, pair_att_28, pair_att_29, pair_att_30], axis=3) # [pos_neg, 7, 7, 256*30]

            # muliply的两个元素形状均为：[pos+neg,7,7,7680], multiply代表逐元素相乘
            # 每个body pair的256个通道的特征，逐项乘以这个pair对应的7*7的attention score，得到新的feature
            part_pairs_att = tf.multiply(pool5_part_pairs, pair_att)
        return part_pairs_att

    def part_pairs_stream(self, part_pairs, is_training, initializer, name):
        with tf.variable_scope(name) as scope:
            # part_pairs = tf.reshape(part_pairs, [tf.shape(part_pairs)[0], 7*7*10752])
            part_pairs = tf.reduce_mean(part_pairs, axis=[1, 2]) # [?, 10752]
            fc6_part_pairs    = slim.fully_connected(part_pairs, 4096, weights_initializer=initializer, trainable=is_training)
            fc6_part_pairs    = slim.dropout(fc6_part_pairs, keep_prob=0.5, is_training=is_training)
            fc7_part_pairs    = slim.fully_connected(fc6_part_pairs, 4096, weights_initializer=initializer, trainable=is_training)
            fc7_part_pairs    = slim.dropout(fc7_part_pairs, keep_prob=0.5, is_training=is_training)

        return fc7_part_pairs

    # 返回形状为 [pos+neg, 600], 为每一个H-O pair的 600个 HOI 动作的可能性
    def region_classification_partpair(self, fc7_part_pairs, is_training, initializer, name):
        with tf.variable_scope(name) as scope:
            cls_score_part_pairs = slim.fully_connected(fc7_part_pairs, self.num_classes,
                                                        weights_initializer=initializer,
                                                        trainable=is_training,
                                                        activation_fn=None, scope='cls_score_part_pairs')
            cls_prob_part_pairs = tf.nn.sigmoid(cls_score_part_pairs, name='cls_prob_part_pairs')

            tf.reshape(cls_prob_part_pairs, [1, self.num_classes])  # [pos+neg,600]

            self.predictions["cls_score_part_pairs"] = cls_score_part_pairs
            self.predictions["cls_prob_part_pairs"] = cls_prob_part_pairs

            self.predictions['cls_prob_PP'] = cls_prob_part_pairs

        return cls_prob_part_pairs

##########################################################################################################
# building the network
##########################################################################################################
    # 构建整个网络
    def build_network(self, is_training):
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        print("######################################################################################")

        # ResNet Backbone
        head     = self.image_to_head(is_training)   # 形状为 (1, ?, ?, 1024)

        ### TIN ###
        sp       = self.sp_to_head()                 # 形状为 (num_pos_neg, 5408)

        # pool5_O的输入只有GT的框的原因是：
        # H、O stream算600类loss的时候是没算negative pair的loss的，因为可能会误监督
        pool5_H  = self.crop_pool_layer(head, self.H_boxes, 'Crop_H')                 # 形状为 (?,7,7,1024),猜测为(num_pos_neg,7,7,1024)
        pool5_O  = self.crop_pool_layer(head, self.O_boxes[:self.H_num,:], 'Crop_O')  # 形状为 (?,7,7,1024),猜测为(num_pos,7,7,1024)

        fc7_H, fc7_O = self.res5_TIN(pool5_H, pool5_O, is_training, 'res5')   # 形状为 (?, 2048)

        # 均代表整幅图片的特征
        head_phi = slim.conv2d(head, 512, [1, 1], scope='head_phi')  # 形状为 (1, ?, ?, 512),猜测为(1,H,W,512)
        head_g   = slim.conv2d(head, 512, [1, 1], scope='head_g')    # 形状为 (1, ?, ?, 512)

        Att_H      = self.attention_pool_layer_H(head_phi, fc7_H, is_training, 'Att_H')    # 形状为 (?,?,?,1),猜测为（num_pos_neg,H,W,1)
        Att_H      = self.attention_norm_H(Att_H, 'Norm_Att_H') # softmax
        att_head_H = tf.multiply(head_g, Att_H)         # 形状为 (?, ?, ?, 512)，猜测为（num_pos_neg, H, W, 512)

        Att_O      = self.attention_pool_layer_O(head_phi, fc7_O, is_training, 'Att_O')
        Att_O      = self.attention_norm_O(Att_O, 'Norm_Att_O') # softmax
        att_head_O = tf.multiply(head_g, Att_O)

        # 在获取 Human, Object的context feature的1024维特征时，在最后的FC层前的最后一个卷积层
        pool5_SH   = self.bottleneck(att_head_H, is_training, 'bottleneck', False)   # 形状为 (?, ?, ?, 1024)，猜测形状是（num_pos_neg, H, W, 1024)
        pool5_SO   = self.bottleneck(att_head_O, is_training, 'bottleneck', True)    # 形状为 (?, ?, ?, 1024)，猜测形状是（num_pos, H, W, 1024)

        # P与C网络生成fc7_SH,fc7_SO的权重共享
        fc9_SH, fc9_SO, fc7_SHsp, fc7_SH, fc7_SO = self.head_to_tail(fc7_H, fc7_O, pool5_SH, pool5_SO, sp, is_training, 'fc_HO')

        # P网络 concat Human Stream, Object Stream, Spatial-pose Stream, 再经过FC，使每个H-O pair拉成1024维张量
        fc9_binary = self.binary_discriminator(fc7_H, fc7_O, fc7_SH, fc7_SO, sp, is_training, 'fc_binary')

        cls_prob_H, cls_prob_O, cls_prob_sp = self.region_classification_TIN(fc9_SH, fc9_SO, fc7_SHsp, is_training, initializer, 'classification_TIN')

        # add a Discriminator here to make binary classification
        cls_prob_binary = self.binary_classification(fc9_binary, is_training, initializer, 'binary_classification')

        self.visualize["attention_map_H"] = (Att_H - tf.reduce_min(Att_H[0,:,:,:])) / tf.reduce_max((Att_H[0,:,:,:] - tf.reduce_min(Att_H[0,:,:,:])))
        self.visualize["attention_map_O"] = (Att_O - tf.reduce_min(Att_O[0,:,:,:])) / tf.reduce_max((Att_O[0,:,:,:] - tf.reduce_min(Att_O[0,:,:,:])))

        ### Partpair ###
        head_part_pair = slim.conv2d(head, 256, [1, 1], scope='head_part_pair')  # (1, H, W, 256)
        # roi of h box, and whole image box
        pool5_H = self.crop_pool_layer(head, self.H_boxes, 'Crop_H')  # (?, 7, 7, 1024)
        pool5_S = self.crop_pool_layer(head, self.S_boxes, 'Crop_S')  # (?, 7, 7, 1024)
        # rois of human go through the res block
        pool5_H = self.res5_partpair(pool5_H, is_training, name='res5_HR')  # (?, 7, 7, 2048)
        pool5_part_pairs = self.ROI_for_part_pair(head_part_pair, self.P_boxes, 'pool5_part_pairs')  # (?, 7, 7, 7680)
        # part pair roi with attention
        part_pairs_att = self.part_pairs_attention(pool5_part_pairs, is_training, initializer,'part_pairs_att')  # (?, 7, 7, 7680)
        # part pairs and h, scene
        part_pairs = tf.concat([pool5_H, pool5_S, part_pairs_att], axis=3)  # [pos_neg, 7, 7, 10752]
        # part pair feature--concat--2fc4096
        fc7_part_pairs = self.part_pairs_stream(part_pairs, is_training=is_training, initializer=initializer,name='part_pairs_fc')  # [pos+neg,4096]
        cls_prob_part_pairs = self.region_classification_partpair(fc7_part_pairs, is_training, initializer, 'classification_partpair')

        # 函数把字典self.predictions的键/值对更新到score_summaries里，有相同键的话直接覆盖
        self.score_summaries.update(self.predictions)

        return cls_prob_H, cls_prob_O, cls_prob_sp, cls_prob_binary, cls_prob_part_pairs


    def create_architecture(self, is_training):
        # 构建整个网络
        cls_prob_H, cls_prob_O, cls_prob_sp, cls_prob_binary, cls_prob_part_pairs = self.build_network(is_training)

        # train summary为包含所有可学习参数的list
        for var in tf.compat.v1.trainable_variables():
            self.train_summaries.append(var)

        # 计算loss，并且将loss的更新写入 layers_to_output 字典，该字典是整个函数的返回变量
        self.add_loss()
        layers_to_output = {}
        layers_to_output.update(self.losses)

        # 创建 Human, Object Stream中的 Attention map以及对应原始图片的 image summary
        tf.compat.v1.summary.image('ATTENTION_MAP_H', self.visualize["attention_map_H"], max_outputs=1)
        tf.compat.v1.summary.image('ATTENTION_MAP_O', self.visualize["attention_map_O"], max_outputs=1)
        self.add_gt_image_summary_H()
        self.add_gt_image_summary_HO()

        # 为 train_summary 这个list中的每一个张量增加一个histogram类型的 summary
        for var in self.train_summaries:
            self.add_train_summary(var)

        # 为所有的 loss 以及 lr 这几个数，分别增加一个 scalar类型的 summary
        for key, var in self.event_summaries.items():
            tf.compat.v1.summary.scalar(key, var)
        tf.compat.v1.summary.scalar('lr', self.lr)

        # merge所有的summary
        self.summary_op     = tf.compat.v1.summary.merge_all()

        return layers_to_output

    # 计算 binary_cross_entropy, H_cross_entropy, O_cross_entropy, sp_cross_entropy
    # 更新 self.losses, event_summaries
    # 返回 loss
    def add_loss(self):
        with tf.compat.v1.variable_scope('LOSS') as scope:
            # Ground Truth
            label_HO     = self.gt_class_HO
            label_binary = self.gt_binary_label

            ### TIN loss ###
            # weight的计算方式：a*lg[1/(k/N)]，其中a是常数，k/N是某一类样本在所有样本中出现的概率（或者说所占的比例）
            cls_score_binary = self.predictions["cls_score_binary"] # here use cls_score, not cls_prob
            cls_score_H  = self.predictions["cls_score_H"]
            cls_score_O  = self.predictions["cls_score_O"]
            cls_score_sp = self.predictions["cls_score_sp"]
            cls_score_H_with_weight = tf.multiply(cls_score_H, self.HO_weight)
            cls_score_O_with_weight = tf.multiply(cls_score_O, self.HO_weight)
            cls_score_sp_with_weight = tf.multiply(cls_score_sp, self.HO_weight)
            cls_score_binary_with_weight = tf.multiply(cls_score_binary, self.binary_weight)
            binary_cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_binary, logits=cls_score_binary_with_weight))
            H_cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_HO[:self.H_num, :],logits=cls_score_H_with_weight[:self.H_num, :]))
            O_cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_HO[:self.H_num, :],logits=cls_score_O_with_weight[:self.H_num,:]))  # fake :self.H_num
            sp_cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_HO, logits=cls_score_sp_with_weight))
            base_loss = H_cross_entropy + O_cross_entropy + sp_cross_entropy

            ### Partpair loss ###
            cls_score_part_pairs = self.predictions["cls_score_part_pairs"]
            ### the HOI contained by gt, give the wts, the others are given wts 1, to enhance the corresponding HOIs' loss
            self.transfer_1_HO = tf.multiply(self.HO_weight, label_HO)  # --> [wts 0 wts 0 0], label is [0 or 1]
            self.transfer_2_HO = tf.subtract(self.transfer_mask_1,label_HO)  # --> [1 1 1 1 1] - [1 0 1 0 0] = [0 1 0 1 1]
            self.transfer_3_HO = tf.add(self.transfer_1_HO,self.transfer_2_HO)  # --> [wts 0 wts 0 0] + [0 1 0 1 1] = [wts 1 wts 1 1], then * loss, element-wise
            part_pairs_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label_HO,logits=cls_score_part_pairs)
            part_pairs_cross_entropy = tf.reduce_mean(tf.multiply(part_pairs_cross_entropy, self.transfer_3_HO))  # self.HO_weight))

            # update self.losses
            self.losses['binary_cross_entropy']     = binary_cross_entropy
            self.losses['H_cross_entropy']          = H_cross_entropy
            self.losses['O_cross_entropy']          = O_cross_entropy
            self.losses['sp_cross_entropy']         = sp_cross_entropy
            self.losses['part_pairs_cross_entropy'] = part_pairs_cross_entropy
            self.losses['base_loss']                = H_cross_entropy + O_cross_entropy + sp_cross_entropy

            # 根据cfg.TRAIN_MODULE参数，定义总的loss
            if cfg.TRAIN_MODULE == 1:
              loss = base_loss + binary_cross_entropy + part_pairs_cross_entropy
            elif cfg.TRAIN_MODULE == 2:
              loss = base_loss + binary_cross_entropy
            elif cfg.TRAIN_MODULE == 3:
              loss = part_pairs_cross_entropy

            self.losses['total_loss'] = loss

            self.event_summaries.update(self.losses)

        return loss

    def add_gt_image_summary_H(self):

        image = tf.py_func(draw_bounding_boxes_HOI, 
                      [tf.reverse(self.image+cfg.PIXEL_MEANS, axis=[-1]), self.H_boxes, self.gt_class_HO],
                      tf.float32, name="gt_boxes_H")
        return tf.compat.v1.summary.image('GROUND_TRUTH_H', image)

    def add_gt_image_summary_HO(self):

        image = tf.py_func(draw_bounding_boxes_HOI, 
                      [tf.reverse(self.image+cfg.PIXEL_MEANS, axis=[-1]), self.O_boxes, self.gt_class_HO],
                      tf.float32, name="gt_boxes_HO")
        return tf.compat.v1.summary.image('GROUND_TRUTH_HO)', image)

    def add_score_summary(self, key, tensor):
        tf.compat.v1.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

    def add_train_summary(self, var):
        tf.compat.v1.summary.histogram('TRAIN/' + var.op.name, var)

    def train_step(self, sess, blobs, lr, train_op):
        feed_dict = {self.image: blobs['image'],
                     self.H_boxes: blobs['H_boxes'], self.O_boxes: blobs['O_boxes'],
                     self.P_boxes: blobs['P_boxes'], self.S_boxes: blobs['S_boxes'],
                     self.gt_class_HO: blobs['gt_class_HO'], self.spatial:blobs['sp'],
                     self.lr: lr, self.H_num: blobs['H_num'], self.gt_binary_label: blobs['binary_label']}
        
        total_loss, base_loss, binary_loss, part_loss, _ = sess.run([self.losses['total_loss'],self.losses['base_loss'],
                                                                    self.losses['binary_cross_entropy'],self.losses['part_pairs_cross_entropy'],train_op],
                                                                    feed_dict=feed_dict)
        return total_loss, base_loss, binary_loss, part_loss

    def train_step_with_summary(self, sess, blobs, lr, train_op):
        feed_dict = {self.image: blobs['image'],
                     self.H_boxes: blobs['H_boxes'], self.O_boxes: blobs['O_boxes'],
                     self.P_boxes: blobs['P_boxes'], self.S_boxes: blobs['S_boxes'],
                     self.gt_class_HO: blobs['gt_class_HO'], self.spatial: blobs['sp'],
                     self.lr: lr, self.H_num: blobs['H_num'], self.gt_binary_label: blobs['binary_label']}

        total_loss, base_loss, binary_loss, part_loss, summary, _ = sess.run(
                                                                        [self.losses['total_loss'],self.losses['base_loss'],
                                                                        self.losses['binary_cross_entropy'],self.losses['part_pairs_cross_entropy'],
                                                                        self.summary_op,train_op],
                                                                        feed_dict=feed_dict)
        return total_loss, base_loss, binary_loss, part_loss, summary

    # return late fusion prediction, cls_prob
    def test_image_HO(self, sess, image, blobs):
        # feed_dict = {self.image: image,
        #              self.H_boxes: blobs['H_boxes'], self.O_boxes: blobs['O_boxes'],
        #              self.P_boxes: np.zeros([1,10,5]), self.S_boxes: np.zeros([1,5]),
        #              self.spatial: blobs['sp'], self.H_num: blobs['H_num']}
        # cls_prob_HO, cls_prob_binary = sess.run([self.predictions["cls_prob_TIN"], self.predictions["cls_prob_binary"]], feed_dict=feed_dict) #self.predictions["cls_prob_binary"]
        #
        # return cls_prob_HO, cls_prob_binary

        feed_dict = {self.image: image,
                     self.H_boxes: blobs['H_boxes'], self.O_boxes: blobs['O_boxes'],
                     self.P_boxes: blobs['P_boxes'], self.S_boxes: blobs['S_boxes'],
                     # self.P_boxes: np.zeros([1, 10, 5]), self.S_boxes: np.zeros([1, 5]),
                     self.spatial: blobs['sp'], self.H_num: blobs['H_num']}
        cls_prob_TIN, cls_prob_binary,cls_prob_PP,cls_prob_H,cls_prob_O,cls_prob_sp = sess.run(
                                                             [self.predictions["cls_prob_TIN"],
                                                              self.predictions["cls_prob_binary"],
                                                              self.predictions["cls_prob_PP"],
                                                              self.predictions["cls_prob_H"],
                                                              self.predictions["cls_prob_O"],
                                                              self.predictions["cls_prob_sp"]],
                                                feed_dict=feed_dict)  # self.predictions["cls_prob_binary"]

        return cls_prob_TIN, cls_prob_binary,cls_prob_PP,cls_prob_H,cls_prob_O,cls_prob_sp
