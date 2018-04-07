import argparse
from collections import namedtuple
import importlib
import logging
import os
import shutil
import sys

import numpy as np
import tensorflow as tf

from pointnet import provider


NUM_CLASSES = 40

Operations = namedtuple(
    'Operations',
    ['pointclouds_pl', 'labels_pl', 'is_training_pl', 'pred',
     'loss', 'train_op', 'merged', 'step']
)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        FLAGS.learning_rate,  # Base learning rate.
                        batch * FLAGS.batch_size,  # Current index into the dataset.
                        FLAGS.decay_step,          # Decay step.
                        FLAGS.decay_rate,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch * FLAGS.batch_size,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        with get_device():
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(FLAGS.batch_size, FLAGS.num_point)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, end_points)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(FLAGS.batch_size)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if FLAGS.optimizer == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=FLAGS.momentum)
            elif FLAGS.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            else:
                raise ValueError('Unknown optimizer %s' % FLAGS.optimizer)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        #sess.run(init)
        sess.run(init, {is_training_pl: True})

        ops = Operations(
            pointclouds_pl=pointclouds_pl, labels_pl=labels_pl,
            is_training_pl=is_training_pl, pred=pred, loss=loss,
            train_op=train_op, merged=merged, step=batch
        )

        for epoch in range(FLAGS.max_epoch):
            logging.info('**** EPOCH %03d ****', epoch)
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)
            
            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(FLAGS.log_dir, "model.ckpt"))
                logging.info("Model saved in file: %s", save_path)


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    num_batches = 0

    for batch_data, batch_labels in data_iter(TEST_FILES):
        # Augment batched point clouds by rotation and jittering
        rotated_data = provider.rotate_point_cloud(batch_data)
        jittered_data = provider.jitter_point_cloud(rotated_data)
        feed_dict = {ops.pointclouds_pl: jittered_data,
                     ops.labels_pl: batch_labels,
                     ops.is_training_pl: is_training,}
        summary, step, _, loss_val, pred_val = sess.run(
            [ops.merged, ops.step, ops.train_op, ops.loss, ops.pred],
            feed_dict=feed_dict
        )
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == batch_labels)
        total_correct += correct
        total_seen += FLAGS.batch_size
        loss_sum += loss_val
        num_batches += 1

    logging.info('mean loss: %f', loss_sum / float(num_batches))
    logging.info('accuracy: %f', total_correct / float(total_seen))


def shuffle_many(*args):
    if not args:
        return []
    shuffle = np.random.permutation(len(args[0]))
    return tuple(np.asanyarray(a)[shuffle] for a in args)


def data_iter(source_files):
    shuffle = np.random.permutation(len(source_files))
    source_files = [source_files[i] for i in shuffle]

    for f in (source_files[i] for i in shuffle):
        logging.info('---- %s -----', f)
        # Load data, and take only the amount of points which is allowed
        current_data, current_label = provider.loadDataFile(f)
        current_data = current_data[:, 0:FLAGS.num_point, :]
        current_label = np.squeeze(current_label)
        # Shuffle the data
        current_data, current_label = shuffle_many((current_data, current_label))
        num_batches = current_data.shape[0] // FLAGS.batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * FLAGS.batch_size
            end_idx = (batch_idx + 1) * FLAGS.batch_size
            yield current_data[start_idx:end_idx], current_label[start_idx:end_idx]


def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0] * NUM_CLASSES
    total_correct_class = [0] * NUM_CLASSES

    for batch_data, batch_labels in data_iter(TEST_FILES):
        feed_dict = {ops.pointclouds_pl: batch_data,
                     ops.labels_pl: batch_labels,
                     ops.is_training_pl: is_training}
        summary, step, loss_val, pred_val = sess.run(
            [ops.merged, ops.step, ops.loss, ops.pred],
            feed_dict=feed_dict
        )
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == batch_labels)
        total_correct += correct
        total_seen += FLAGS.batch_size
        loss_sum += (loss_val*FLAGS.batch_size)
        for l, p in zip(batch_labels, pred_val):
            total_seen_class[l] += 1
            total_correct_class[l] += (p == l)
            
    logging.info('eval mean loss: %f', loss_sum / float(total_seen))
    logging.info('eval accuracy: %f', total_correct / float(total_seen))
    logging.info('eval avg class acc: %f', np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)))
         

def init_logging():
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")

    root_logger = logging.getLogger()

    file_handler = logging.FileHandler(filename='log_train.txt', mode='w')
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)


def get_device():
    if FLAGS.gpu >= 0:
        return tf.device('/device:GPU:%d' % FLAGS.gpu)
    else:
        return tf.device('/cpu:0')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to use [default: GPU 0]')
    parser.add_argument('--model', default='pointnet_cls',
                        choices=['pointnet_cls', 'pointnet_cls_basic', 'pointnet_seg'],
                        help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
    parser.add_argument('--log_dir', default='log',
                        help='Log dir [default: log]')
    parser.add_argument('--num_point', type=int, default=1024,
                        choices=[256, 512, 1024, 2048],
                        help='Point Number [256/512/1024/2048] [default: 1024]')
    parser.add_argument('--max_epoch', type=int, default=250,
                        help='Epoch to run [default: 250]')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch Size during training [default: 32]')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate [default: 0.001]')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum to use with momentum optimizer [default: 0.9]')
    parser.add_argument('--optimizer', default='adam',
                        choices=['adam', 'momentum'],
                        help='adam or momentum [default: adam]')
    parser.add_argument('--decay_step', type=int, default=200000,
                        help='Decay step for lr decay [default: 200000]')
    parser.add_argument('--decay_rate', type=float, default=0.7,
                        help='Decay rate for lr decay [default: 0.8]')
    
    FLAGS = parser.parse_args()

    BN_INIT_DECAY = 0.5
    BN_DECAY_DECAY_RATE = 0.5
    BN_DECAY_DECAY_STEP = float(FLAGS.decay_step)
    BN_DECAY_CLIP = 0.99

    init_logging()

    logging.info(FLAGS)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    MODEL = importlib.import_module(FLAGS.model)  # import network module
    MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model + '.py')

    os.makedirs(FLAGS.log_dir, exist_ok=True)
    # bkp of model def
    shutil.copy(MODEL_FILE, FLAGS.log_dir)
    # bkp of train procedure
    shutil.copy('train.py', FLAGS.log_dir)

    DATA_DIR = os.path.join(BASE_DIR, 'data')
    provider.download_modelnet40(DATA_DIR)
    # ModelNet40 official train/test split
    TRAIN_FILES = provider.getDataFiles(
        os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'train_files.txt')
    )
    TEST_FILES = provider.getDataFiles(
        os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'test_files.txt')
    )

    train()
