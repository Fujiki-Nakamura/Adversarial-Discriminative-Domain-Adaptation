# coding: UTF-8
from __future__ import print_function

import argparse
from collections import deque
import logging
import os

import keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import tensorflow as tf
from tqdm import tqdm

from models import source_cnn
import util


def main(args):
    util.config_logging()
    # Parameters
    lr = args.lr
    batch_size = args.batch_size
    iterations = args.iterations
    snapshot = args.snapshot
    stepsize = args.stepsize
    display = args.display
    output_dir = 'output/source_cnn/'
    output_dir_clf = 'output/classifier/'
    save_path = os.path.join(output_dir, 'source_cnn.ckpt')
    save_path_clf = os.path.join(output_dir_clf, 'classifier.ckpt')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir_clf):
        os.makedirs(output_dir_clf)

    # Load train data
    train_mat = loadmat('./data/svhn/train_32x32.mat')
    train_images = train_mat['X'].transpose((3, 0, 1, 2))
    train_images = train_images.astype(np.float32) / 255.
    train_labels = train_mat['y'].squeeze()
    train_labels[train_labels == 10] = 0
    RGB2GRAY = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
    train_images = np.sum(
        np.multiply(train_images, RGB2GRAY),
        3, keepdims=True
    )
    # Data generator
    idg = ImageDataGenerator()
    train_data_gen = idg.flow(
        train_images, train_labels, batch_size=batch_size, shuffle=True
    )

    # Define graph
    nb_classes = 10
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    x_resized = tf.image.resize_images(x, [28, 28])
    t = tf.placeholder(tf.int32, (None,))
    logits = source_cnn(x_resized, nb_classes=nb_classes, trainable=True)
    loss = tf.losses.sparse_softmax_cross_entropy(t, logits)
    lr_var = tf.Variable(lr, name='learning_rate', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss)

    t_vars = tf.trainable_variables()
    source_cnn_vars = [
        var for var in t_vars
        if var.name.startswith('source_cnn')
    ]
    classifier_vars = [
        var for var in t_vars
        if var.name.startswith('classifier')
    ]

    # Train
    cnn_saver = tf.train.Saver(var_list=source_cnn_vars)
    if len(classifier_vars) > 0:
        clf_saver = tf.train.Saver(var_list=classifier_vars)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    losses = deque(maxlen=10)
    training_losses = []
    bar = tqdm(range(iterations))
    bar.set_description('(lr: {:.0e})'.format(lr))
    bar.refresh()

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i in bar:
            batch_images, batch_labels = next(train_data_gen)
            loss_val, _ = sess.run(
                [loss, train_op], feed_dict={x: batch_images, t: batch_labels}
            )
            losses.append(loss_val)
            training_losses.append(loss_val)
            if i % display == 0:
                logging.info('{:20} {:10.4f}     (avg: {:10.4f})'.format(
                    'Iteration {}:'.format(i), loss_val, np.mean(losses)))
            if stepsize is not None and (i + 1) % stepsize == 0:
                lr = sess.run(lr_var.assign(lr * 0.1))
                logging.info('Changed learning rate to {:.0e}'.format(lr))
                bar.set_description('(lr: {:.0e})'.format(lr))
            if (i + 1) % snapshot == 0:
                snapshot_path = cnn_saver.save(sess, save_path)
                if len(classifier_vars) > 0:
                    clf_saver.save(sess, save_path_clf)
                logging.info('Saved snapshot to {}'.format(snapshot_path))

    plt.plot(training_losses, label='Source CNN Loss')
    plt.title('Pre-Training Loss')
    plt.legend()
    plt.savefig('./pre_training_losses.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--display', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--snapshot', type=int, default=5000)
    parser.add_argument('--stepsize', type=int, default=None)
    args = parser.parse_args()
    main(args)
