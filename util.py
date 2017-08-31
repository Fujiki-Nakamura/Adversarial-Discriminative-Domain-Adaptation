from collections import OrderedDict
from functools import reduce
import logging
import logging.config
import gzip
import operator
import os.path
import struct

import numpy as np
import tensorflow as tf
import yaml

from tqdm import tqdm


def _read_datafile(path, expected_dims):
    """Helper function to read a file in IDX format."""
    base_magic_num = 2048
    with gzip.GzipFile(path) as f:
        magic_num = struct.unpack('>I', f.read(4))[0]
        expected_magic_num = base_magic_num + expected_dims
        if magic_num != expected_magic_num:
            raise ValueError('Incorrect MNIST magic number (expected '
                             '{}, got {})'
                             .format(expected_magic_num, magic_num))
        dims = struct.unpack('>' + 'I' * expected_dims,
                             f.read(4 * expected_dims))
        buf = f.read(reduce(operator.mul, dims))
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(*dims)
        return data

def _read_images(path):
    """Read an MNIST image file."""
    return (_read_datafile(path, 3)
            .astype(np.float32)
            .reshape(-1, 28, 28, 1)
            / 255)

def _read_labels(path):
    """Read an MNIST label file."""
    return _read_datafile(path, 1)

class TqdmHandler(logging.StreamHandler):

    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


def config_logging(logfile=None):
    path = os.path.join(os.path.dirname(__file__), 'logging.yml')
    with open(path, 'r') as f:
        config = yaml.load(f.read())
    if logfile is None:
        del config['handlers']['file_handler']
        del config['root']['handlers'][-1]
    else:
        config['handlers']['file_handler']['filename'] = logfile
    logging.config.dictConfig(config)


def remove_first_scope(name):
    return '/'.join(name.split('/')[1:])

def collect_vars(scope, start=None, end=None, prepend_scope=None):
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    var_dict = OrderedDict()
    if isinstance(start, str):
        for i, var in enumerate(vars):
            var_name = remove_first_scope(var.op.name)
            if var_name.startswith(start):
                start = i
                break
    if isinstance(end, str):
        for i, var in enumerate(vars):
            var_name = remove_first_scope(var.op.name)
            if var_name.startswith(end):
                end = i
                break
    for var in vars[start:end]:
        var_name = remove_first_scope(var.op.name)
        if prepend_scope is not None:
            var_name = os.path.join(prepend_scope, var_name)
        var_dict[var_name] = var
    return var_dict
