# --------------------------------------------------------
# Tensorflow TIN
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ult.config import cfg
from ult.ult import Get_Next_Instance_HO_Neg_HICO_pose_pattern_version2
from ult.timer import Timer

import pickle
import numpy as np
import os
import sys
import glob
import time
import ipdb

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.training.learning_rate_decay import cosine_decay_restarts
import os

 
class SolverWrapper(object):
    """
    A wrapper class for the training process
    """

    def __init__(self, sess, network, Trainval_GT, Trainval_N, output_dir, tbdir, Pos_augment, Neg_select, Restore_flag, pretrained_model, interval_divide):

        self.net               = network
        self.Trainval_GT       = self.changeForm(Trainval_GT, interval_divide)
        self.Trainval_N        = Trainval_N
        self.output_dir        = output_dir
        self.tbdir             = tbdir
        self.Pos_augment       = Pos_augment
        self.Neg_select        = Neg_select
        self.Restore_flag      = Restore_flag
        self.pretrained_model  = pretrained_model

    # 每到达一定的iteration，用saver.save保存模型参数
    def snapshot(self, sess, iter, total_loss, base_loss, binary_loss, part_loss):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Store the model snapshot
        filename = 'HOI' + '_iter_{:d}'.format(iter) + '.ckpt'
        filename = os.path.join(self.output_dir, filename)
        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

        file = cfg.ROOT_DIR+'/Weights/losses4.txt'
        with open(file, "a+") as f:
            f.write('{:<12} {:.4f} {:.4f} {:.4f},{:.4f}\n'.format(iter, total_loss, base_loss, binary_loss, part_loss))


    # 将Traval_GT变换形式，同一幅图片的proposals按照5个聚成新的GT项
    # 新的GT项从原来的一维数组（数组每一项为一组H-O proposal）
    # 变成二维数组（即原来一维H-O proposal数组，至多5个一组组成的数组，只有相同图片的proposal才能放到一起
    def changeForm(self, Trainval_GT, interval_divide):
        GT_dict = {}
        for item in Trainval_GT:
            try:
                GT_dict[item[0]].append(item)
            except KeyError:
                GT_dict[item[0]] = [item]

        GT_new = []
        for image_id, value in GT_dict.items():
            count = 0
            length = len(value)
            while count < length:
                temp = value[count: min(count + interval_divide, length)]
                count += len(temp)
                GT_new.append(temp)

        return GT_new


    def construct_graph(self, sess):
        with sess.graph.as_default():
            # Set the random seed for tensorflow
            tf.compat.v1.set_random_seed(cfg.RNG_SEED)
            # Build the main computation graph
            layers = self.net.create_architecture(True) # is_training flag: True
            # Define the loss
            loss = layers['total_loss']

            # 获取 global_step
            if cfg.TRAIN_MODULE_CONTINUE == 1:  # from iter_ckpt
                path_iter = self.pretrained_model.split('.ckpt')[0]
                iter_num = path_iter.split('_')[-1]
                global_step    = tf.Variable(int(iter_num), trainable=False)
            elif cfg.TRAIN_MODULE_CONTINUE == 2:  # from iter 0
                global_step    = tf.Variable(0, trainable=False)

            # 根据 global_step 获取 lr
            # 指数下降的 lr
            # lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE * 10, global_step, cfg.TRAIN.STEPSIZE * 5, cfg.TRAIN.GAMMA, staircase=True)
            # 余弦下降的 lr
            first_decay_steps = 80000  # first_decay_steps 是指第一次完全下降的 step 数
            t_mul, m_mul = 2.0, 1.0    # t_mul 是指每一次循环的步数都将乘以 t_mul 倍, # m_mul 指每一次循环重新开始时的初始 lr 是上一次循环初始值的 m_mul 倍
            lr = cosine_decay_restarts(cfg.TRAIN.LEARNING_RATE * 10, global_step, first_decay_steps, t_mul, m_mul, alpha=0.0)

            # 定义优化函数，使用Momentum算法的Optimizer
            self.optimizer = tf.compat.v1.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)

            # 1--Update_all_parameter, 2--Only_Update_D, 3--Update_H+O+SP, 4--updating except classifiers of S(fc)
            list_var_to_update = []
            if cfg.TRAIN_MODULE_UPDATE == 1:
                list_var_to_update = tf.compat.v1.trainable_variables()
            if cfg.TRAIN_MODULE_UPDATE == 2:
                list_var_to_update = [var for var in tf.compat.v1.trainable_variables() if 'fc_binary' in var.name or 'binary_classification' in var.name]
            if cfg.TRAIN_MODULE_UPDATE == 3:
                list_var_to_update = [var for var in tf.compat.v1.trainable_variables() if 'fc_binary' not in var.name or 'binary_classification' not in var.name]
            if cfg.TRAIN_MODULE_UPDATE == 4:
                list_var_to_update = [var for var in tf.compat.v1.trainable_variables() if 'classification' not in var.name]

            # 计算各个train_variable的梯度
            grads_and_vars = self.optimizer.compute_gradients(loss, list_var_to_update)
            capped_gvs     = [(tf.clip_by_norm(grad, 1.), var) for grad, var in grads_and_vars]  # 将梯度的L2范数缩放为 1, 防止梯度爆炸
            # 将计算出的梯度应用到变量上
            train_op = self.optimizer.apply_gradients(capped_gvs,global_step=global_step)

            # 新建一个 Saver 用于保存模型，这个Saver在snapshot()中用到
            # max_to_keep用来控制检查点的数量，超过max_to_keep后，新来一个checkpoint将删除原有的旧的一个，我们程序里max_to_keep=None
            self.saver = tf.compat.v1.train.Saver(max_to_keep=cfg.TRAIN.SNAPSHOT_KEPT)
            # 新建一个 Writer 用于将训练时的summary写入tensorboard
            self.writer = tf.compat.v1.summary.FileWriter(self.tbdir, sess.graph)

        return lr, train_op


    def from_snapshot(self, sess):
        if self.Restore_flag == 0:
            saver_t  = [var for var in tf.compat.v1.model_variables() if 'conv1' in var.name and 'conv1_sp' not in var.name]
            saver_t += [var for var in tf.compat.v1.model_variables() if 'conv2' in var.name and 'conv2_sp' not in var.name]
            saver_t += [var for var in tf.compat.v1.model_variables() if 'conv3' in var.name]
            saver_t += [var for var in tf.compat.v1.model_variables() if 'conv4' in var.name]
            saver_t += [var for var in tf.compat.v1.model_variables() if 'conv5' in var.name]
            saver_t += [var for var in tf.compat.v1.model_variables() if 'shortcut' in var.name]

            sess.run(tf.compat.v1.global_variables_initializer())

            ###############################################################################################
            for var in tf.compat.v1.trainable_variables():
                print(var.name, var.eval().mean())

            print('Restoring model snapshots from {:s}'.format(self.pretrained_model))

            self.saver_restore = tf.compat.v1.train.Saver(saver_t)
            self.saver_restore.restore(sess, self.pretrained_model)

            for var in tf.compat.v1.trainable_variables():
                print(var.name, var.eval().mean())

        if self.Restore_flag == 5 or self.Restore_flag == 6 or self.Restore_flag == 7:
            sess.run(tf.compat.v1.global_variables_initializer())

            # print()
            # print("before")
            # for var in tf.compat.v1.trainable_variables():
            #     print(var.name, var.eval().mean())

            print('Restoring model snapshots from {:s}'.format(self.pretrained_model))
            saver_t = {}

            # Add block0
            for ele in tf.compat.v1.model_variables():
                if 'resnet_v1_50/conv1/weights' in ele.name or 'resnet_v1_50/conv1/BatchNorm/beta' in ele.name or 'resnet_v1_50/conv1/BatchNorm/gamma' in ele.name or 'resnet_v1_50/conv1/BatchNorm/moving_mean' in ele.name or 'resnet_v1_50/conv1/BatchNorm/moving_variance' in ele.name:
                    # ele.name如：     resnet_v1_50/conv1/weights:0
                    # ele.name[:-2]如：resnet_v1_50/conv1/weights
                    saver_t[ele.name[:-2]] = ele
            # Add block1
            for ele in tf.compat.v1.model_variables():
                if 'block1' in ele.name:
                    saver_t[ele.name[:-2]] = ele
            # Add block2
            for ele in tf.compat.v1.model_variables():
                if 'block2' in ele.name:
                    saver_t[ele.name[:-2]] = ele
            # Add block3
            for ele in tf.compat.v1.model_variables():
                if 'block3' in ele.name:
                    saver_t[ele.name[:-2]] = ele
            # Add block4
            for ele in tf.compat.v1.model_variables():
                if 'block4' in ele.name:
                    saver_t[ele.name[:-2]] = ele
            
            self.saver_restore = tf.compat.v1.train.Saver(saver_t)
            self.saver_restore.restore(sess, self.pretrained_model)

            # 用 pretrained model中的resnet的block4来初始化我们模型的 resnet 的 block5
            # 这时候在我们打开的sess里，我们的model的 block4和 block5被初始化为相同的值
            if self.Restore_flag >= 5:
                saver_t = {}
                # Add block5
                for ele in tf.compat.v1.model_variables():
                    if 'block4' in ele.name:
                        saver_t[ele.name[:-2]] = [var for var in tf.compat.v1.model_variables() if ele.name[:-2].replace('block4','block5') in var.name][0]

                self.saver_restore = tf.compat.v1.train.Saver(saver_t)
                self.saver_restore.restore(sess, self.pretrained_model)

            if self.Restore_flag >= 6:
                saver_t = {}
                # Add block6
                for ele in tf.compat.v1.model_variables():
                    if 'block4' in ele.name:
                        saver_t[ele.name[:-2]] = [var for var in tf.compat.v1.model_variables() if ele.name[:-2].replace('block4','block6') in var.name][0]

                self.saver_restore = tf.compat.v1.train.Saver(saver_t)
                self.saver_restore.restore(sess, self.pretrained_model)
                
            if self.Restore_flag >= 7:
                saver_t = {}
                # Add block7
                for ele in tf.compat.v1.model_variables():
                    if 'block4' in ele.name:
                        saver_t[ele.name[:-2]] = [var for var in tf.compat.v1.model_variables() if ele.name[:-2].replace('block4','block7') in var.name][0]

                self.saver_restore = tf.compat.v1.train.Saver(saver_t)
                self.saver_restore.restore(sess, self.pretrained_model)

        print("*************************************************************************************")
        print("trainable_variables: ")
        for var in tf.compat.v1.trainable_variables():
            print(var.name, var.eval().mean())


    def from_previous_ckpt(self,sess):

        sess.run(tf.compat.v1.global_variables_initializer())
        for var in tf.compat.v1.trainable_variables(): # trainable weights, we need surgery
            print(var.name, var.eval().mean())

        print('Restoring model snapshots from {:s}'.format(self.pretrained_model))
        saver_t = {}

        saver_t  = [var for var in tf.compat.v1.model_variables() if 'fc_binary' not in var.name \
                                                       and 'binary_classification' not in var.name \
                                                       and 'conv1_pose_map' not in var.name \
                                                       and 'pool1_pose_map' not in var.name \
                                                       and 'conv2_pose_map' not in var.name \
                                                       and 'pool2_pose_map' not in var.name]

        self.saver_restore = tf.compat.v1.train.Saver(saver_t)
        self.saver_restore.restore(sess, self.pretrained_model)

        print("the variables is being trained now \n")
        for var in tf.compat.v1.trainable_variables():
           print(var.name, var.eval().mean())

    
    def from_best_trained_model(self, sess):

        sess.run(tf.compat.v1.global_variables_initializer())
        for var in tf.compat.v1.trainable_variables(): # trainable weights, we need surgery
            print(var.name, var.eval().mean())

        print('Restoring model snapshots from {:s}'.format(self.pretrained_model))
        saver_t = {}

        saver_t  = [var for var in tf.compat.v1.model_variables() if 'fc_binary' not in var.name \
                                           and 'binary_classification' not in var.name \
                                           and 'conv1_pose_map' not in var.name \
                                           and 'pool1_pose_map' not in var.name \
                                           and 'conv2_pose_map' not in var.name \
                                           and 'pool2_pose_map' not in var.name]

        for var in tf.compat.v1.trainable_variables():
            print(var.name, var.eval().mean())

        # for ele in tf.compat.v1.model_variables():
        #     saver_t[ele.name[:-2]] = ele

        self.saver_restore = tf.compat.v1.train.Saver(saver_t)
        self.saver_restore.restore(sess, self.pretrained_model)


        print("the variables is being trained now \n")
        for var in tf.compat.v1.trainable_variables():
           print(var.name, var.eval().mean())


    def train_model(self, sess, max_iters):
        timer = Timer()
        Data_length = len(self.Trainval_GT)
        lr, train_op = self.construct_graph(sess)

        # 加载初始的模型参数
        if cfg.TRAIN_MODULE_CONTINUE == 1:  # continue training
            self.from_previous_ckpt(sess)
        else:                               # from iter 0 ,默认是这个
            # Initializing weight: 1--from faster RCNN  2--from previous best  3--from our model with d
            if cfg.TRAIN_INIT_WEIGHT == 2:
                self.from_best_trained_model(sess)
            elif cfg.TRAIN_INIT_WEIGHT == 1:
                self.from_snapshot(sess)
            elif cfg.TRAIN_INIT_WEIGHT == 3:  # load all paras including D, initial from our best
                self.from_previous_ckpt(sess)

        # 将图变为只读(read-only)，新的操作就不能够添加到图里了
        sess.graph.finalize()

        # 获取模型当前的iter值
        if cfg.TRAIN_MODULE_CONTINUE == 2:   # from iter 0 ,默认是这个
            iter = 0
        elif cfg.TRAIN_MODULE_CONTINUE == 1: # from iter_ckpt
            path_iter = self.pretrained_model.split('.ckpt')[0]
            iter_num = path_iter.split('_')[-1]
            iter = int(iter_num)

        cur_min = 10
        # 执行max_iters次梯度迭代
        while iter < max_iters + 1:
            timer.tic()
            # 获取增强后的一张图片的信息
            blobs = Get_Next_Instance_HO_Neg_HICO_pose_pattern_version2(self.Trainval_GT, self.Trainval_N, iter, self.Pos_augment, self.Neg_select, Data_length)

            # 执行一次梯度下降
            # train_step_with_summary传入lr是为了记录lr的summary
            if (iter % cfg.TRAIN.SUMMARY_INTERVAL == 0) or (iter < 20):   # Compute the graph with summary
                total_loss, base_loss, binary_loss, part_loss, summary = self.net.train_step_with_summary(sess, blobs, lr.eval(), train_op)
                self.writer.add_summary(summary, float(iter))
            else:                                                         # Compute the graph without summary
                total_loss, base_loss, binary_loss, part_loss = self.net.train_step(sess, blobs, lr.eval(), train_op)
            timer.toc()

            # 打印训练信息
            if iter % (cfg.TRAIN.DISPLAY) == 0:
                print('iter: %d / %d, im_id: %u, lr: %f, speed: %.3f s/iter\ntotal  loss: %.6f\nbase   loss: %.6f\nbinary loss: %.6f\npart   loss: %.6f' % \
                      (iter, max_iters, self.Trainval_GT[iter%Data_length][0][0], lr.eval(), timer.average_time, total_loss, base_loss, binary_loss, part_loss))
            # 保存模型
            if (iter % cfg.TRAIN.SNAPSHOT_ITERS * 5 == 0 and iter != 0) or (iter == 10) or (iter > 1000 and total_loss<cur_min-0.0001):
                if (iter > 1000 and total_loss<cur_min-0.0001):
                    cur_min=total_loss
                self.snapshot(sess, iter, total_loss, base_loss, binary_loss, part_loss)
            # 更新迭代器
            iter += 1

        self.writer.close()


def train_net(network, Trainval_GT, Trainval_N, output_dir, tb_dir, Pos_augment, Neg_select, Restore_flag, pretrained_model, max_iters=300000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)

    # 当cfg.TRAIN_MODULE_CONTINUE == 2，意味着模型从头开始训练，移除所有此前保存过的模型和 logs
    # if cfg.TRAIN_MODULE_CONTINUE == 2:     # training from iter 0
    #     # Remove previous events
    #     filelist = [ f for f in os.listdir(tb_dir)]
    #     for f in filelist:
    #         os.remove(os.path.join(tb_dir, f))
    #     # Remove previous snapshots
    #     filelist = [ f for f in os.listdir(output_dir)]
    #     for f in filelist:
    #         os.remove(os.path.join(output_dir, f))
        
    tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=True)   # 允许tf自动选择一个存在并且可用的设备来运行操作
    tfconfig.gpu_options.allow_growth = True               # 刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放

    interval_divide = 5

    with tf.compat.v1.Session(config=tfconfig) as sess:              # 无需sess.close()，运行完毕后自己释放资源
        sw = SolverWrapper(sess, network, Trainval_GT, Trainval_N, output_dir, tb_dir, Pos_augment, Neg_select, Restore_flag, pretrained_model, interval_divide)
        print('Solving..., Pos augment = ' + str(Pos_augment) + ', Neg augment = ' + str(Neg_select) + ', Restore_flag = ' + str(Restore_flag))
        sw.train_model(sess, max_iters)
        print('done solving')
