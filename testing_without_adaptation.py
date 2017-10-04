# coding: UTF-8
import numpy as np
import tensorflow as tf

import models
import util


def main():
    # parameters
    path_source_cnn = './output/source_cnn'
    nb_classes = 10
    batch_size = 128

    # Load target data
    target_images = util._read_images(
        './data/mnist/train-images-idx3-ubyte.gz')
    assert 0.0 <= np.min(target_images) and np.max(target_images) <= 1.0
    target_labels = util._read_labels(
        './data/mnist/train-labels-idx1-ubyte.gz')

    # Graphs
    tf.reset_default_graph()
    x_target = tf.placeholder(tf.float32, (None, 28, 28, 1))
    logits = models.source_cnn(x_target, nb_classes)
    prediction = tf.argmax(logits, axis=1)
    src_vars = [
        var for var in tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='source_cnn')]
    source_saver = tf.train.Saver(var_list=src_vars)

    # Test
    preds = []
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        source_saver.restore(
            sess,
            tf.train.latest_checkpoint(path_source_cnn)
        )
        n_data = len(target_images)
        for start_i in range(0, n_data, batch_size):
            end_i = start_i + batch_size
            X_batch = target_images[start_i:end_i]
            pred = sess.run(prediction, {x_target: X_batch})
            preds.extend(pred)

    assert len(preds) == n_data
    n_correct = np.equal(preds, target_labels).sum()
    acc = n_correct / float(n_data)
    print('acc = {}'.format(acc))


if __name__ == '__main__':
    main()
