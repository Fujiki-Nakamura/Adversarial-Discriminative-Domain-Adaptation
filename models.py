# coding: UTF-8
import tensorflow as tf


def classifier(net, nb_classes, trainable=True):
    with tf.variable_scope('classifier'):
        # fc4
        logits = tf.layers.dense(
            net, nb_classes, trainable=trainable
        )

    return logits


def source_cnn(x, nb_classes, trainable=False, adapt=False):
    with tf.variable_scope('source_cnn'):
        padding = 'valid'
        initializer=tf.contrib.layers.xavier_initializer()
        # conv1
        net = tf.layers.conv2d(
            x, filters=20, kernel_size=5, strides=(1, 1),
            padding=padding,
            trainable=trainable,
            kernel_initializer=initializer,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5)
        )
        net = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net, pool_size=2, strides=2)
        # conv2
        net = tf.layers.conv2d(
            net, filters=50, kernel_size=5, strides=(1, 1),
            padding=padding,
            trainable=trainable,
            kernel_initializer=initializer,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5)
        )
        net = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net, pool_size=2, strides=2)
        # fc3
        net = tf.contrib.layers.flatten(net)
        net = tf.layers.dense(
            net, 500, trainable=trainable,
            kernel_initializer=initializer,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5)
        )
        net = tf.nn.relu(net)
        # fc4
        net = tf.layers.dense(
            net, nb_classes, trainable=trainable,
            kernel_initializer=initializer,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5)
        )

    return net


def target_cnn(x, nb_classes, trainable=True, training=True, testing=False):
    with tf.variable_scope('target_cnn'):
        padding = 'valid'
        initializer=tf.contrib.layers.xavier_initializer()
        # conv1
        net = tf.layers.conv2d(
            x, filters=20, kernel_size=5, strides=(1, 1),
            padding=padding,
            trainable=trainable,
            kernel_initializer=initializer,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5)
        )
        net = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net, pool_size=2, strides=2)
        # conv2
        net = tf.layers.conv2d(
            net, filters=50, kernel_size=5, strides=(1, 1),
            padding=padding,
            trainable=trainable,
            kernel_initializer=initializer,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5)
        )
        net = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net, pool_size=2, strides=2)
        # fc3
        net = tf.contrib.layers.flatten(net)
        net = tf.layers.dense(
            net, 500, trainable=trainable,
            kernel_initializer=initializer,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5)
        )
        net = tf.nn.relu(net)
        # fc4
        net = tf.layers.dense(
            net, nb_classes, trainable=trainable,
            kernel_initializer=initializer,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5)
        )

    return net


def discriminator(feature, reuse=False, alpha=0.2):
    with tf.variable_scope('discriminator', reuse=reuse):
        initializer=tf.contrib.layers.xavier_initializer()
        # fc1
        net = tf.layers.dense(
            feature, 500, activation=None,
            kernel_initializer=initializer,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5)
        )
        net = tf.nn.relu(net)
        # net = tf.maximum(alpha * net, net)
        # fc2
        net = tf.layers.dense(
            net, 500, activation=None,
            kernel_initializer=initializer,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5)
        )
        net = tf.nn.relu(net)
        # net = tf.maximum(alpha * net, net)
        # output
        net = tf.layers.dense(
            net, 1, activation=None,
            kernel_initializer=initializer,
        )

    return net
