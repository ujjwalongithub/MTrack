import tensorflow as tf
import os
import glob
import argparse
import configparser
import typing
import pandas as pd
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


def get_data_folder(mot_folder: str,  subset: str) -> typing.List[str]:
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
