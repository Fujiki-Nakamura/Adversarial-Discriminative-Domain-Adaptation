# coding: UTF-8
from __future__ import print_function

import argparse
from collections import deque
import logging
import os

from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import tensorflow as tf
from tqdm import tqdm

from models import discriminator
from models import source_cnn
from models import target_cnn
import util


def main(args):
    util.config_logging()
    # Settings
    lr = args.lr
    beta1 = args.beta1
    batch_size = args.batch_size
    iterations = args.iterations
    snapshot = args.snapshot
    stepsize = args.stepsize
    display = args.display

    path_source_cnn = './output/source_cnn'
    output_dir = os.path.join('output', 'target_cnn')
    save_path = os.path.join(output_dir, 'target_cnn.ckpt')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load source data
    train_mat = loadmat('./data/svhn/train_32x32.mat')
    train_images = train_mat['X'].transpose((3, 0, 1, 2))
    train_images = train_images.astype(np.float32) / 255.
    RGB2GRAY = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
    train_images = np.sum(
        np.multiply(train_images, RGB2GRAY),
        3, keepdims=True
    )
    assert 0.0 <= np.min(train_images) and np.max(train_images) <= 1.0
    # Load target data
    target_images = util._read_images(
        './data/mnist/train-images-idx3-ubyte.gz')
    assert 0.0 <= np.min(target_images) and np.max(target_images) <= 1.0
    # Data generator
    idg = ImageDataGenerator()
    source_data_gen = idg.flow(
        train_images, batch_size=batch_size, shuffle=True
    )
    target_data_gen = idg.flow(
        target_images, batch_size=batch_size, shuffle=True
    )

    # Define graph
    nb_classes = 10
    tf.reset_default_graph()
    x_source = tf.placeholder(tf.float32, (None, 32, 32, 1))
    x_source_resized = tf.image.resize_images(x_source, [28, 28])
    x_target = tf.placeholder(tf.float32, (None, 28, 28, 1))

    feature_src = source_cnn(
        x_source_resized, nb_classes=nb_classes,
        trainable=False, adapt=True)
    feature_target = target_cnn(x_target, nb_classes, trainable=True)
    d_logits_src = discriminator(feature_src)
    d_logits_target = discriminator(feature_target, reuse=True)

    # Loss: Discriminator
    d_loss_src = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logits_src, labels=tf.ones_like(d_logits_src)))
    d_loss_target = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logits_target, labels=tf.zeros_like(d_logits_target)))
    d_loss = d_loss_src + d_loss_target
    # Loss: target CNN
    target_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logits_target, labels=tf.ones_like(d_logits_target)))

    t_vars = tf.trainable_variables()
    target_vars = [var for var in tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope='target_cnn')]
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    src_vars = [
        var for var in tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='source_cnn')]

    lr_var = tf.Variable(lr, name='learning_rate', trainable=False)
    optimizer = tf.train.AdamOptimizer(lr_var, beta1)
    target_train_op = optimizer.minimize(target_loss, var_list=target_vars)
    d_train_op = optimizer.minimize(d_loss, var_list=d_vars)

    # Train
    source_saver = tf.train.Saver(var_list=src_vars)
    target_saver = tf.train.Saver(var_list=target_vars)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    target_losses = deque(maxlen=10)
    d_losses = deque(maxlen=10)
    bar = tqdm(range(iterations))
    bar.set_description('(lr: {:.0e})'.format(lr))
    bar.refresh()
    losses = []

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        source_saver.restore(
            sess,
            tf.train.latest_checkpoint(path_source_cnn)
        )
        for i in bar:
            batch_source = next(source_data_gen)
            batch_target = next(target_data_gen)

            target_loss_val, d_loss_val, _, _ = sess.run(
                [target_loss, d_loss, target_train_op, d_train_op],
                feed_dict={x_source: batch_source, x_target: batch_target}
            )
            target_losses.append(target_loss_val)
            d_losses.append(d_loss_val)
            losses.append([target_loss_val, d_loss_val])
            if i % display == 0:
                logging.info('{:20} Target: {:5.4f} (avg: {:5.4f})'
                             '    Discriminator: {:5.4f} (avg: {:5.4f})'
                             .format('Iteration {}:'.format(i),
                                     target_loss_val,
                                     np.mean(target_losses),
                                     d_loss_val,
                                     np.mean(d_losses)))
            if stepsize is not None and (i + 1) % stepsize == 0:
                lr = sess.run(lr_var.assign(lr * 0.1))
                logging.info('Changed learning rate to {:.0e}'.format(lr))
                bar.set_description('(lr: {:.0e})'.format(lr))
            if (i + 1) % snapshot == 0:
                snapshot_path = target_saver.save(sess, save_path)
                logging.info('Saved snapshot to {}'.format(snapshot_path))

    # Save visualization of training losses
    losses = np.array(losses)
    plt.plot(losses.T[0], label='Target CNN Loss', alpha=0.5)
    plt.plot(losses.T[1], label='Discriminator Loss', alpha=0.5)
    plt.title('Training Losses')
    plt.legend()
    os.remove('./losses.png')
    plt.savefig('./losses.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--display', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--snapshot', type=int, default=5000)
    parser.add_argument('--stepsize', type=int, default=None)
    parser.add_argument('--beta1', type=float, default=0.5)
    args = parser.parse_args()
    main(args)
