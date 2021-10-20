import argparse
import configparser
import glob
import io
import itertools
import os
import typing

import cv2
import more_itertools as mit
import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
import multiprocessing as mp

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

parser.add_argument('--tfrecord_dir', type=str, help='Location to save the '
                                                     'TFRecord files to.')


def get_data_folders(mot_folder: str, subset: str) -> typing.List[str]:
    subset_folder = 'MOT17-{}'.format(subset)
    root_data_folder = os.path.join(mot_folder, subset_folder)
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


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


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
    return boxes, ids, to_pad


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

    # df = df[df.class_id == 1]
    #
    # logger.debug('Number of rows after retaining only pedestrians = {'
    #              '}.'.format(df.shape[0]))

    df_grp = df.groupby(['frame_id'], sort=True)
    return df_grp


def get_shard_filename(subset: str, shard_num: int, num_shards: int) -> str:
    return 'MOT17-{}-{}-of-{}'.format(subset, str(shard_num).zfill(4),
                                      str(num_shards).zfill(4))





def main():
    args = parser.parse_args()
    mot_dir = args.mot_dir
    subset = args.subset
    num_frames = args.num_frames
    num_shards = args.num_shards
    tfrecord_dir = args.tfrecord_dir
    os.makedirs(tfrecord_dir, exist_ok=True)
    tfrecord_names = [
        get_shard_filename(subset, i, num_shards) for i in range(1,
                                                                 num_shards + 1)
    ]

    tfrecord_names = list(
        map(
            lambda x : os.path.join(tfrecord_dir, x),
            tfrecord_names
        )
    )

    data_folders = get_data_folders(mot_dir, subset)
    image_chunks = list()
    bounding_boxes = list()
    tracking_ids = list()
    sequence_information = list()
    paddings = list()
    for folder in data_folders:
        seq_info = get_mot_seq_info(folder)
        image_list = get_images(folder)
        image_list = chunk_images(image_list, num_frames)
        image_chunks.append(image_list)
        gt_grp = read_gt(folder)
        sequence_information.append([seq_info] * len(image_list))
        for img_list in image_list:
            boxes, ids, padding = get_gt_for_batch(img_list, gt_grp)
            bounding_boxes.append(boxes)
            tracking_ids.append(ids)
            paddings.append(padding)

    image_chunks = list(itertools.chain(*image_chunks))
    sequence_information = list(itertools.chain(*sequence_information))

    image_chunks = list(
        list(c) for c in
        mit.divide(num_shards, image_chunks)
    )

    sequence_information = list(
        list(c) for c in
        mit.divide(num_shards, sequence_information)
    )

    bounding_boxes = list(
        list(c) for c in
        mit.divide(num_shards, bounding_boxes)
    )

    tracking_ids = list(
        list(c) for c in
        mit.divide(num_shards, tracking_ids)
    )

    paddings = list(
        list(c) for c in
        mit.divide(num_shards, paddings)
    )


    pool_args = list(
        zip(
            tfrecord_names,
            image_chunks,
            bounding_boxes,
            tracking_ids,
            paddings,
            sequence_information
        )
    )

    pool = mp.Pool(processes=mp.cpu_count())
    pool.starmap(create_one_shard, pool_args)
    return None


def read_frames_as_bytes(frame_files):
    images = list()
    for imagename in frame_files:
        image = cv2.imread(imagename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)

    images = np.stack(images, axis=0)
    images = images.tobytes()
    return images


def create_example(frame_files, boxes, tracking_ids, paddings, sequence_info):
    frame_bytes = read_frames_as_bytes(frame_files)
    image_width = int(sequence_info['imWidth'])
    image_height = int(sequence_info['imHeight'])
    total_boxes_with_paddings = boxes.shape[0]
    ymins = boxes[:, :, 0]/ image_height
    xmins = boxes[:, :, 1] / image_width
    ymaxs = boxes[:, :, 2]/ image_height
    xmaxs = boxes[:, :, 3]/ image_width
    ymins = np.reshape(ymins, -1).tolist()
    xmins = np.reshape(xmins, -1).tolist()
    ymaxs = np.reshape(ymaxs, -1).tolist()
    xmaxs = np.reshape(xmaxs, -1).tolist()
    tracking_ids = np.reshape(tracking_ids, -1).tolist()
    tracking_ids = list(map(int, tracking_ids))
    features = tf.train.Features(
        feature={
            'image/height': int64_feature(image_height),
            'image/width': int64_feature(image_width),
            'image/encoded': bytes_feature(frame_bytes),
            'xmins': float_list_feature(xmins),
            'ymins': float_list_feature(ymins),
            'xmaxs': float_list_feature(xmaxs),
            'ymaxs': float_list_feature(ymaxs),
            'tracking_ids': int64_list_feature(tracking_ids),
            'paddings': int64_list_feature(paddings),
            'total_boxes_with_paddings': int64_feature(
                total_boxes_with_paddings)
        }
    )

    tf_example = tf.train.Example(
        features=features
    )
    return tf_example


def create_one_shard(tfrecord_name, image_chunks, boxes, tracking_ids,
                     paddings, sequence_information):
    writer = tf.io.TFRecordWriter(tfrecord_name)

    for i in range(len(image_chunks)):
        tf_example = create_example(image_chunks[i], boxes[i], tracking_ids[
            i], paddings[i], sequence_information[i])
        writer.write(tf_example.SerializeToString())

    logger.info('Wrote {} records to {}.'.format(len(image_chunks),
                                                 tfrecord_name))
    writer.close()
    return None


if __name__ == "__main__":
    main()
