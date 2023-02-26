from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import glob, os
print('Imported libraries')

DS_PATH = 'dataset'
IMGS_PER_FILE = 4096
os.makedirs(f'{DS_PATH}/tfrecs', exist_ok=True)

captions = open(f'{DS_PATH}/id_to_text.txt').read().strip().split('\n')
img_fnames = sorted(glob.glob(f'{DS_PATH}/imgs/*'))

def strip_non_ascii(text):
    stripped = (c for c in text if 0 < ord(c) < 127)
    return ''.join(stripped)

def image_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value.numpy()])
    )

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

def create_example(image, caption):
    feature = {
        'image': image_feature(image),
        'caption': bytes_feature(caption),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

AUTO = tf.data.experimental.AUTOTUNE

def parse_tfrecord_fn(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'caption': tf.io.FixedLenFeature([],tf.string),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example['image'] = tf.io.decode_jpeg(example['image'], channels=3)
    return example

dec = tf.io.decode_jpeg
enc = tf.io.encode_jpeg

'''
# example loading dataset
raw_dataset = tf.data.TFRecordDataset(glob.glob(f'{DS_PATH}/tfrecs/*tfrec'))
parsed_dataset = raw_dataset.map(parse_tfrecord_fn, AUTO).batch(256)
for idx, i in enumerate(parsed_dataset):
    print(i['caption'].shape, i['image'].shape)
'''

for idx, (img_fname, caption) in tqdm(enumerate(zip(img_fnames, captions))):
    if idx % IMGS_PER_FILE == 0:
        filename = f'{DS_PATH}/tfrecs/{(idx//IMGS_PER_FILE):0>5}.tfrec'
        writer = tf.io.TFRecordWriter(filename)
    img_jpg = tf.io.read_file(img_fname)
    example_proto = create_example(img_jpg, strip_non_ascii(caption))
    #print(parse_tfrecord_fn(example_proto.SerializeToString())['image'].shape)
    writer.write(example_proto.SerializeToString())
