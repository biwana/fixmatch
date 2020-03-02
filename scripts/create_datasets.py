#!/usr/bin/env python

# Copyright 2018 Google LLC
# modified by Brian K. Iwana
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to download all datasets and create .tfrecord files.
"""

import collections
import gzip
import os
import tarfile
import tempfile
from urllib import request
from PIL import Image

import numpy as np
import scipy.io
import tensorflow as tf
from absl import app
from tqdm import trange, tqdm

from libml import data as libml_data
from libml.utils import EasyDict

def _load_fold_0():
    return _load_fold(0)

def _load_fold_1():
    return _load_fold(1)

def _load_fold_2():
    return _load_fold(2)

def _load_fold_3():
    return _load_fold(3)

def _load_fold_4():
    return _load_fold(4)

def _load_fold(fold):

    def get_labeled_files(root_folder):
        clas = []
        files = []
        for cla in os.listdir(root_folder):
            for file in os.listdir(os.path.join(root_folder, cla)):
                if file.endswith(".png"):
                    files.append(file)
                    clas.append(cla)

        files = np.reshape(files, (-1, 1))
        clas = np.reshape(clas, (-1, 1))
        y = np.hstack((files, clas))
        np.random.shuffle(y)
        return y
    
    def get_unlabeled_files(root_folder):
        files = []
        for file in os.listdir(root_folder):
            if file.endswith(".png"):
                files.append(file)

        y = np.reshape(files, (-1, 1))
        np.random.shuffle(y)
        return y
    
    def load_images(root_folder, files):
        images = np.zeros((files.shape[0], input_shape[0], input_shape[1], input_shape[2]))
        if files.shape[1] == 2:
            for it, f in enumerate(tqdm(files[:,0])):
                img = Image.open(os.path.join(root_folder, str(files[it,1]), f))
                img.load()
                images[it] = np.asarray(img, dtype="int32")[:,:,:3]
        else:
            for it, f in enumerate(tqdm(files)):
                img = Image.open(os.path.join(root_folder, f[0]))
                img.load()
                images[it] = np.asarray(img, dtype="int32")[:,:,:3]
        return images
    
    root_train = os.path.join("data","train","fold_%d"%fold)
    l_train = get_labeled_files(root_train)
    np.savetxt(os.path.join("data", "train_fold_%d.txt"%fold), l_train, fmt="%s %s")

    root_test = os.path.join("data","test","fold_%d"%fold)
    l_test = get_unlabeled_files(root_test)
    np.savetxt(os.path.join("data", "test_fold_%d.txt"%fold), l_test, fmt="%s")
    
    root_unsupervised_train = os.path.join("data","unsupervised_train","_")
    l_unsupervised_train = get_unlabeled_files(root_unsupervised_train)
    np.savetxt(os.path.join("data", "unsupervised_train%d.txt"%fold), l_test, fmt="%s")
    
    input_shape=(224,224,3)

    x_train = load_images(root_train, l_train)
    x_test = load_images(root_test, l_test)
    x_unsupervised_train = load_images(root_unsupervised_train, l_unsupervised_train)
    
    train_set = {'images': x_train,
                 'labels': l_train[:,1].astype(np.uint8)}

    test_set = {'images': x_test,
                'labels': np.zeros(x_test.shape[0], dtype=np.uint8)}

    unlabeled_set = {'images': x_unsupervised_train,
                     'labels': np.zeros(x_unsupervised_train.shape[0], dtype=np.uint8)}

    train_set['images'] = _encode_png(train_set['images'])
    test_set['images'] = _encode_png(test_set['images'])
    unlabeled_set['images'] = _encode_png(unlabeled_set['images'])
    return dict(train=train_set, test=test_set, unlabeled=unlabeled_set)

def _encode_png(images):
    raw = []
    with tf.Session() as sess, tf.device('cpu:0'):
        image_x = tf.placeholder(tf.uint8, [None, None, None], 'image_x')
        to_png = tf.image.encode_png(image_x)
        for x in trange(images.shape[0], desc='PNG Encoding', leave=False):
            raw.append(sess.run(to_png, feed_dict={image_x: images[x]}))
    return raw


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _save_as_tfrecord(data, filename):
    assert len(data['images']) == len(data['labels'])
    filename = os.path.join(libml_data.DATA_DIR, filename + '.tfrecord')
    print('Saving dataset:', filename)
    with tf.python_io.TFRecordWriter(filename) as writer:
        for x in trange(len(data['images']), desc='Building records'):
            feat = dict(image=_bytes_feature(data['images'][x]),
                        label=_int64_feature(data['labels'][x]))
            record = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(record.SerializeToString())
    print('Saved:', filename)


def _is_installed(name, checksums):
    for subset, checksum in checksums.items():
        filename = os.path.join(libml_data.DATA_DIR, '%s-%s.tfrecord' % (name, subset))
        if not tf.gfile.Exists(filename):
            return False
    return True


def _save_files(files, *args, **kwargs):
    del args, kwargs
    for folder in frozenset(os.path.dirname(x) for x in files):
        tf.gfile.MakeDirs(os.path.join(libml_data.DATA_DIR, folder))
    for filename, contents in files.items():
        with tf.gfile.Open(os.path.join(libml_data.DATA_DIR, filename), 'w') as f:
            f.write(contents)


def _is_installed_folder(name, folder):
    return tf.gfile.Exists(os.path.join(libml_data.DATA_DIR, name, folder))


CONFIGS = dict(
    fold_0=dict(loader=_load_fold_0, checksums=dict(train=None, test=None, unlabeledset=None)),
    fold_1=dict(loader=_load_fold_1, checksums=dict(train=None, test=None, unlabeledset=None)),
    fold_2=dict(loader=_load_fold_2, checksums=dict(train=None, test=None, unlabeledset=None)),
    fold_3=dict(loader=_load_fold_3, checksums=dict(train=None, test=None, unlabeledset=None)),
    fold_4=dict(loader=_load_fold_4, checksums=dict(train=None, test=None, unlabeledset=None)),
)


def main(argv):
    if len(argv[1:]):
        subset = set(argv[1:])
    else:
        subset = set(CONFIGS.keys())
    tf.gfile.MakeDirs(libml_data.DATA_DIR)
    for name, config in CONFIGS.items():
        if name not in subset:
            continue
        if 'is_installed' in config:
            if config['is_installed']():
                print('Skipping already installed:', name)
                continue
        elif _is_installed(name, config['checksums']):
            print('Skipping already installed:', name)
            continue
        print('Preparing', name)
        datas = config['loader']()
        saver = config.get('saver', _save_as_tfrecord)
        for sub_name, data in datas.items():
            if sub_name == 'readme':
                filename = os.path.join(libml_data.DATA_DIR, '%s-%s.txt' % (name, sub_name))
                with tf.gfile.Open(filename, 'w') as f:
                    f.write(data)
            elif sub_name == 'files':
                for file_and_data in data:
                    path = os.path.join(libml_data.DATA_DIR, file_and_data.filename)
                    with tf.gfile.Open(path, "wb") as f:
                        f.write(file_and_data.data)
            else:
                saver(data, '%s-%s' % (name, sub_name))


if __name__ == '__main__':
    app.run(main)
