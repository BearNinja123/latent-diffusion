import os; os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from transformers import CLIPTokenizer
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import glob, os, torch
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

tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
AUTO = tf.data.experimental.AUTOTUNE

@tf.function(jit_compile=True)
def scale_nchw(img):
    return tf.transpose(tf.cast(img , 'float32') / 127.5 - 1.0, (2, 0, 1))

dec = tf.io.decode_jpeg
enc = tf.io.encode_jpeg

def parse_tfrecord_fn(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'caption': tf.io.FixedLenFeature([],tf.string),
    }
    example = tf.io.parse_single_example(example, feature_description)
    img = tf.io.decode_jpeg(example['image'])
    example['image'] = scale_nchw(img)
    #example['caption_tokens'] = tokenizer(str(example['caption']), truncation=True, max_length=77, return_length=True, return_overflowing_tokens=False, padding="max_length")['input_ids']

    return example

def benchmark(dataset, num_epochs=2, time_per_step=1.4):
    start_time = time.perf_counter()
    ds_iter = dataset.as_numpy_iterator()
    for epoch_num in range(num_epochs):
        #for idx, data in enumerate(dataset): # before: ~77% flops ratio
        #    if idx == 2:
        #        break
        #    imgs = torch.from_numpy(data['image'].numpy())
        #    imgs = imgs.permute(0, 3, 1, 2).contiguous().float() / 127.5 - 1.0
        #    captions = list(data['caption'].numpy().astype(str))
        #g = dataset.as_numpy_iterator()
        for idx, data in enumerate(ds_iter): # after: ~90% flops ratio
            if idx == 2:
                break
            imgs = torch.from_numpy(data['image'])
            captions = list(data['caption'].astype(str))
            print(idx)
            time.sleep(time_per_step)
    duration = time.perf_counter() - start_time
    print("Execution time:", duration, "ratio max flops:", num_epochs*2*time_per_step/duration)

import time
raw_dataset = tf.data.TFRecordDataset(glob.glob(f'{DS_PATH}/tfrecs/*tfrec'))
raw_dataset = raw_dataset.take(10240)
parsed_dataset = raw_dataset.shuffle(65536).map(parse_tfrecord_fn, AUTO).batch(512).prefetch(AUTO)
benchmark(parsed_dataset)
print('Start')
benchmark(parsed_dataset)
