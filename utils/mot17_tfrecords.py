import tensorflow as tf
import tensorflow_io as tfio
import os
import glob
import argparse
import configparser
import typing

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


@tf.function
def read_video(video_file_name: str):
    file_content = tf.io.read_file(
        filename=video_file_name,
        name='video_reader'
    )
    return tfio.experimental.ffmpeg.decode_video(
        content=file_content,
        index=0,
        name='video_decoder'
    )
