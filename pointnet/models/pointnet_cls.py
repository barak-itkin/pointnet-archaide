import tensorflow as tf
import numpy as np
from pointnet.utils import tf_util
from .transform_nets import input_transform_net, feature_transform_net


def placeholder_inputs(batch_size, num_point, K=3):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, K))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


def get_model_features(point_cloud, is_training, bn_decay=None, K=3,
                       input_transformer=True, feature_transformer=True,
                       reduce_max=False, name_suffix='', skip_last=False):
    """ Classification PointNet, input is BxNxK, output Bx40 """
    end_points = {}

    if input_transformer:
        with tf.variable_scope('transform_net1') as sc:
            transform = input_transform_net(point_cloud, is_training, bn_decay, K=K)
        point_cloud_transformed = tf.matmul(point_cloud, transform)
        # Shape the pointcloud as an image with one channel (append an axis)
        # BxNxKx1
        input_image = tf.expand_dims(point_cloud_transformed, -1)
    else:
        input_image = tf.expand_dims(point_cloud, -1)

    # BxNx1x64
    net = tf_util.conv2d(input_image, 64, [1,K],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1' + name_suffix, bn_decay=bn_decay)
    # BxNx1x64
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2' + name_suffix, bn_decay=bn_decay)

    if feature_transformer:
        with tf.variable_scope('transform_net2') as sc:
            transform = feature_transform_net(net, is_training, bn_decay, K=64)
        end_points['transform'] = transform
        # BxNx64
        net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
        # BxNx1x64
        net_transformed = tf.expand_dims(net_transformed, [2])
    else:
        net_transformed = net

    # BxNx1x64
    net = tf_util.conv2d(net_transformed, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3' + name_suffix, bn_decay=bn_decay)
    # BxNx1x128
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4' + name_suffix, bn_decay=bn_decay)
    # BxNx1x1024
    if not skip_last:
        net = tf_util.conv2d(net, 1024, [1,1],
                             padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training,
                             scope='conv5' + name_suffix, bn_decay=bn_decay)

    # BxNx1024
    net = tf.squeeze(net, axis=2)
    # Symmetric function: max pooling
    # Bx1024
    if reduce_max:
        net = tf.reduce_max(net, axis=1)

    return net, end_points


def get_model_multi_features(point_cloud, names, Ks, is_training, bn_decay=None,
                             input_transformer=True, feature_transformer=True):
    vals = []  # Each value is BxNx1024
    skip = 0
    for name, K in zip(names, Ks):
        val, _ = get_model_features(
            point_cloud=point_cloud[:, :, skip:(skip + K)], K=K, is_training=is_training, bn_decay=bn_decay,
            input_transformer=input_transformer, feature_transformer=feature_transformer,
            reduce_max=False, name_suffix='_' + name, skip_last=True
        )
        vals.append(val)
        skip += K

    net = tf.concat(vals, axis=1)
    # Each value is BxNx1x...
    net = tf.expand_dims(net, axis=2)
    # BxNx1x1024
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=False, is_training=is_training,
                         scope='conv_combine1', bn_decay=bn_decay)

    net = tf.squeeze(net, axis=2)
    net = tf.reduce_max(net, axis=1)
    return net


def get_model_scores(model_features, is_training, n_classes, bn_decay=None):
    net = tf_util.fully_connected(model_features, 512, bn=False, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=False, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')
    return tf_util.fully_connected(net, n_classes, activation_fn=None, scope='fc3')


def get_model(point_cloud, is_training, n_classes, bn_decay=None, K=3,
              input_transformer=True, feature_transformer=True):
    """ Classification PointNet, input is BxNxK, output Bx40 """
    model_features, end_points = get_model_features(
        point_cloud, is_training, bn_decay=bn_decay, K=K,
        input_transformer=input_transformer,
        feature_transformer=feature_transformer
    )
    scores = get_model_scores(model_features, is_training, n_classes, bn_decay)
    return scores, end_points


def get_transform_loss(end_points, reg_weight=0.001):
    if 'transform' in end_points:
        transform = end_points['transform']  # BxKxK
        K = transform.get_shape()[1].value
        mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0, 2, 1]))
        mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
        mat_diff_loss = tf.nn.l2_loss(mat_diff)
    else:
        mat_diff_loss = tf.constant(0, dtype=tf.float32)
    tf.summary.scalar('mat loss', mat_diff_loss)
    return mat_diff_loss * reg_weight


def get_loss(pred, label, end_points, reg_weight=0.0001):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    return classify_loss + get_transform_loss(end_points, reg_weight)


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
