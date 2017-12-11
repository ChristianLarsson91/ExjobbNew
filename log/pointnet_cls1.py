import tensorflow as tf
import numpy as np
import math
import sys
import os
import random
import pdb
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
from transform_nets import input_transform_net, feature_transform_net

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    netStructure=[]
    numberOfTransforms = 0
    convNumber=1
    maxpoolNumber=0
    fcNumber=0
    dropoutNumber=0
    numberOfTransforms = 0

    net = point_cloud

    if random.randint(1,2) == 1:
        layer=["expand",0,0]
        netStructure.append(layer)
        while True:
            i=random.randint(1,3)
            if i == 1:
                convNumber +=1
                layer=["conv2d",math.pow(2,random.randint(4,10)),convNumber]
                netStructure.append(layer)
            elif i == 2:
                maxpoolNumber += 1
                layer=["maxpool",0,maxpoolNumber]
                netStructure.append(layer)
            elif i == 3:
                layer=["transform",0,1]
                netStructure.append(layer)
                break
            else:
                pass
    else:
        layer=["transform",0,1]
        netStructure.append(layer)
    if random.randint(1,2) == 1:
        convNumber +=1
        layer=["conv2d_trans1",math.pow(2,random.randint(4,10)),convNumber]
        netStructure.append(layer)
        while True:
            i=random.randint(1,3)
            if i == 1:
                convNumber +=1
                layer=["conv2d",math.pow(2,random.randint(4,10)),convNumber]
                netStructure.append(layer)
            elif i == 2:
                maxpoolNumber += 1
                layer=["maxpool",0,maxpoolNumber]
                netStructure.append(layer)
            elif i == 3:
                    convNumber +=1
                    layer=["conv2d",64,convNumber]
                    netStructure.append(layer)
                    layer=["transform",0,2]
                    netStructure.append(layer)
                    convNumber +=1
                    layer=["conv2d",math.pow(2,random.randint(4,10)),convNumber]
                    netStructure.append(layer)
                    break
            else:
                pass
    else:
        convNumber +=1
        layer=["conv2d",64,convNumber]
        netStructure.append(layer)
        layer=["transform",0,2]
        netStructure.append(layer)
        convNumber +=1
        layer=["conv2d_trans2",math.pow(2,random.randint(4,10)),convNumber]
        netStructure.append(layer)

    while True:
        i=random.randint(1,3)
        if i == 1:
            convNumber +=1
            layer=["conv2d",math.pow(2,random.randint(4,10)),convNumber]
            netStructure.append(layer)
        elif i == 2:
            maxpoolNumber += 1
            layer=["maxpool",0,maxpoolNumber]
            netStructure.append(layer)
        elif i == 3:
            fcNumber += 1
            layer=["fc",math.pow(2,random.randint(4,10)),fcNumber]
            netStructure.append(layer)
            break
        else:
            pass
    while True:
        i=random.randint(1,3)
        if i==1:
            fcNumber += 1
            layer=["fc",math.pow(2,random.randint(4,10)),fcNumber]
            netStructure.append(layer)
        elif i==2:
            dropoutNumber += 1
            layer=["dropout",0,dropoutNumber]
            netStructure.append(layer)
            fcNumber += 1
            layer=["fc",math.pow(2,random.randint(4,10)),fcNumber]
            netStructure.append(layer)
        elif i==3:
            fcNumber += 1
            layer=["fc",4,fcNumber]
            netStructure.append(layer)
            break
        else:
            pass
    print(netStructure)
    netStructure = [["transform",0,1],["conv2d",64,1],["transform",0,2],["conv2d",32,2],["maxpool",0,1],["fc",64,1],["dropout",0,1],["fc",4,2]]
    for layer in netStructure:

        print(layer)
        if layer[0] == "conv2d":
            print("conv")
            if layer[2] == 1:
                net = tf_util.conv2d(input_image,layer[1],[1,3],padding='VALID',stride=[1,1],
                bn=True, is_training=is_training, scope='conv%d'%(layer[2]), bn_decay=bn_decay)
            else:
                net = tf_util.conv2d(net, layer[1], [1,1],
            padding='VALID', stride=[1,1],
            bn=True, is_training=is_training,
            scope='conv%d'%(layer[2]), bn_decay=bn_decay)
            print(layer[:])
        elif layer[0] == "maxpool":
            net = tf_util.max_pool2d(net, [num_point,1],padding='VALID', scope='maxpool%d'%(layer[2]))
            print(layer[:])
        elif layer[0] == "transform":
            if layer[2]==1:
                with tf.variable_scope('transform_net%d' %(layer[2])) as sc:
                    transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
                point_cloud_transformed = tf.matmul(point_cloud, transform)
                input_image = tf.expand_dims(point_cloud_transformed, -1)
                pdb.set_trace()
            else:
                with tf.variable_scope('transform_net%d' %(layer[2])) as sc:
                    transform = feature_transform_net(net, is_training, bn_decay, K=64)
                end_points['transform'] = transform
                net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
                net_transformed = tf.expand_dims(net_transformed, [2])
                pdb.set_trace()


        elif layer[0] == "fc":
            if layer [2] == 1:
                pdb.set_trace()
                net = tf.reshape(net, [batch_size, -1])
                net = tf_util.fully_connected(net, layer[1], bn=True, is_training=is_training, scope='fc%d'%(layer[2]), bn_decay = bn_decay)

            elif layer[1] == 4:
                net = tf_util.fully_connected(net, 4, activation_fn=None, scope='fc%d'%(layer[2]))
            else:
                net = tf_util.fully_connected(net, layer[1], bn=True, is_training=is_training,scope='fc%d'%(layer[2]), bn_decay=bn_decay)
            print(layer[:])
        elif layer[0] == "dropout":
            net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,scope='dp%d'%(layer[2]))
            print(layer[:])
        elif layer[0] == "expand":
            print("expand")
            net = tf.expand_dims(net, -1)
        elif layer[0] == "conv2d_trans1":
            net = tf_util.conv2d(input_image, layer[1], [1,3],
            padding='VALID', stride=[1,1],
            bn=True, is_training=is_training,
            scope='conv%d'%(layer[2]), bn_decay=bn_decay)
        elif layer[0] == "conv2d_trans2":
            net = tf_util.conv2d(net_transformed, layer[1], [1, 1],
            padding = 'VALID', stride = [1, 1],
            bn = True, is_training = is_training,
            scope = 'conv%d'%(layer[2]), bn_decay = bn_decay)
        else:
            pass

    return netStructure, net, end_points


def get_loss(pred, label, end_points, reg_weight=0.001):
    """ pred: B*NUM_CLASSES,
        label: B, """
    pdb.set_trace()
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform'] # BxKxK
    K = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff)
    tf.summary.scalar('mat loss', mat_diff_loss)

    return classify_loss + mat_diff_loss * reg_weight


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
