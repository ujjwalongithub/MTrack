import argparse
import configparser
import glob
import os
import typing

import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger

parser = argparse.ArgumentParser(prog='Code to create TFRecords for MOT17.')
parser.add_argument('--mot_dir', help='Path storing MOT17 dataset.')
parser.add_argument('--subset', choices=['train', 'val', 'test'], help='Subset '
                                                                       'of '
                                                                       'MOT17 '
                                                                       'dataset \
                                                                                                                        for which '
                                                                       'the '
                                                                       'TFRecords '
                                                                       'are '
                                                                       'created.')
parser.add_argument('--num_frames', type=int, default=4, help='Number of '
                                                              'frames in each batch.')

parser.add_argument('--num_shards', type=int, default=64, help='Number of '
                                                               'shards.')


def get_data_folders(mot_folder: str, subset: str) -> typing.List[str]:
    root_data_folder = os.path.join(mot_folder, subset)
    return glob.glob(
        os.path.join(
            root_data_folder,
            '*SDP'
        )
    )


def get_mot_seq_info(data_folder: str) -> typing.Dict:
    config = configparser.ConfigParser()
    ini_file_name = os.path.join(data_folder, 'seqinfo.ini')
    config.read(ini_file_name)
    seq_info = dict()
    seq_info['frameRate'] = config['Sequence']['frameRate']
    seq_info['seqLength'] = config['Sequence']['seqLength']
    seq_info['imWidth'] = config['Sequence']['imWidth']
    seq_info['imHeight'] = config['Sequence']['imHeight']
    seq_info['imExt'] = config['Sequence']['imExt']
    return seq_info


@tf.function
def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


@tf.function
def _bytes_list_feature(values):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def get_images(seq_folder: str) -> typing.List[str]:
    img_folder = os.path.join(
        seq_folder,
        'img1'
    )
    img_extension = get_mot_seq_info(seq_folder)['imExt']

    images = glob.glob(
        os.path.join(
            img_folder,
            '*{}'.format(img_extension)
        )
    )

    images.sort(
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )

    return images


def frame_number_from_filename(filename):
    return int(os.path.splitext(os.path.basename(filename))[0])


def get_gt_for_batch(image_list: typing.List[str], gt_grp):
    boxes = list()
    ids = list()
    for imgname in image_list:
        framenum = frame_number_from_filename(imgname)
        gt_df = gt_grp.get_group(framenum)
        bboxes = np.array(
            [
                gt_df['ymin'],
                gt_df['xmin'],
                gt_df['height'],
                gt_df['width']
            ]
        ).transpose()
        bboxes[:, 2] += (bboxes[:, 0] - 1)
        bboxes[:, 3] += (bboxes[:, 1] - 1)
        boxes.append(bboxes)
        tracking_ids = np.array(gt_df['tracking_id'])
        ids.append(tracking_ids)

    max_num = max(
        [
            x.shape[0] for x in boxes
        ]
    )

    to_pad = [
        max_num - x.shape[0] for x in boxes
    ]

    for i in range(len(to_pad)):
        if to_pad[i] == 0:
            continue
        boxes[i] = np.concatenate(
            [
                boxes[i],
                np.zeros((to_pad[i], 4))
            ],
            axis=0
        )
        ids[i] = np.concatenate(
            [
                ids[i],
                np.ones((to_pad[i],)) * -1
            ]
        )

    boxes = np.stack(
        boxes
    )

    ids = np.stack(ids)
    return boxes, ids


def chunk_images(image_list: typing.List[str], num_frames: int) -> \
        typing.List[typing.List[
            str]]:
    chunks = [image_list[i:i + num_frames] for i in range(0, len(image_list),
                                                          num_frames)]

    for chunk in chunks:
        if len(chunk) != num_frames:
            chunks.remove(chunk)

    return chunks


def read_gt(seq_folder: str):
    gt_file = os.path.join(
        seq_folder,
        'gt', 'gt.txt'
    )

    df = pd.read_csv(
        gt_file,
        sep=',',
        names=['frame_id', 'tracking_id', 'xmin', 'ymin', 'width', 'height',
               'ignore_zero', 'class_id', 'confidence'],
        engine='c'
    )

    logger.debug('Initial number of rows = {}.'.format(df.shape[0]))
    df = df[df.ignore_zero != 0]

    logger.debug('Number of rows after ignoring bad GT = {}.'.format(
        df.shape[0]))

    df = df[df.tracking_id >= 0]

    logger.debug('Number of rows after removing negative tracking IDs = {'
                 '}.'.format(df.shape[0]))

    df = df[df.class_id == 1]

    logger.debug('Number of rows after retaining only pedestrians = {'
                 '}.'.format(df.shape[0]))

    df_grp = df.groupby(['frame_id'], sort=True)
    return df_grp


def get_shard_filename(subset: str, shard_num: int, num_shards: int) -> str:
    return 'MOT17-{}-{}-of-{}'.format(subset, str(shard_num).zfill(4),
                                      str(num_shards).zfill(4))


def read_data_folder(data_folder: str, num_frames: int):
    seq_info = get_mot_seq_info(data_folder)
    gt_grp = read_gt(data_folder)
    images = get_images(data_folder)
    image_chunks = chunk_images(images, num_frames)


def main():
    args = parser.parse_args()
    mot_dir = args.mot_dir
    subset = args.subset
    num_frames = args.num_frames
    num_shards = args.num_shards
    tfrecord_names = [
        get_shard_filename(subset, i, num_shards) for i in range(1,
                                                                 num_shards + 1)
    ]

    data_folders = get_data_folders(mot_dir, subset)
    pass
