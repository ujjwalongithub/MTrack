import errno
import glob
import os


def prepare_mot17_trackeval(mot17_folder):
    if not os.path.isdir(mot17_folder):
        raise NotADirectoryError('The folder {} was not found.'.format(
            mot17_folder))

    if os.path.isdir(
        os.path.join(
            mot17_folder,
            'train'
        )
    ):
        os.rename(
            os.path.join(
                mot17_folder,
                'train'
            ),
            os.path.join(
                mot17_folder,
                'MOT17-train'
            )
        )

    if os.path.isdir(
        os.path.join(
            mot17_folder,
            'test'
        )
    ):
        os.rename(
            os.path.join(
                mot17_folder,
                'test'
            ),
            os.path.join(
                mot17_folder,
                'MOT17-test'
            )
        )

    os.makedirs(
        os.path.join(
            mot17_folder,
            'MOT17-val'
        ),
        exist_ok=True
    )

    try:
        os.symlink(
            os.path.join(
                mot17_folder,
                'MOT17-train',
                'MOT17-01-SDP'
            ),
            os.path.join(
                mot17_folder,
                'MOT17-val',
                'MOT17-01-SDP'
            )
        )
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass

    train_seqs = get_train_seq(mot17_folder, root_only=True)

    val_seqs = ['MOT17-01-SDP']

    train_seq_filename = os.path.join(
        mot17_folder, 'MOT17-train.txt'
    )
    with open(train_seq_filename, 'w') as fid:
        for seq in train_seqs:
            fid.write('{}\n'.format(seq))

    val_seq_filename = os.path.join(
        mot17_folder, 'MOT17-val.txt'
    )
    with open(val_seq_filename, 'w') as fid:
        for seq in val_seqs:
            fid.write('{}\n'.format(seq))

    test_seq_filename = os.path.join(
        mot17_folder, 'MOT17-test.txt'
    )

    test_seq = get_test_seq(mot17_folder, root_only=True)

    with open(test_seq_filename, 'w') as fid:
        for seq in test_seq:
            fid.write('{}\n'.format(seq))

    return None


def get_train_seq(mot17_folder, root_only=False):
    train_folder = os.path.join(mot17_folder, 'MOT17-train')
    train_seqs = glob.glob(os.path.join(train_folder, '*SDP'))
    if root_only:
        train_seqs = list(
            map(
                lambda x: os.path.basename(x),
                train_seqs
            )
        )

    if root_only:
        train_seqs = list(set(train_seqs).difference(set(('MOT17-01-SDP'))))
    else:
        train_seqs = list(
            set(
                train_seqs
            ).difference(
                set(
                    (
                        os.path.join(mot17_folder, 'MOT17-train',
                                     'MOT17-01-SDP')
                    )
                )
            )
        )
    return train_seqs


def get_test_seq(mot17_folder, root_only=False):
    test_folder = os.path.join(mot17_folder, 'MOT17-test')
    test_seqs = glob.glob(os.path.join(test_folder, '*SDP'))
    if root_only:
        test_seqs = list(
            map(
                lambda x: os.path.basename(x),
                test_seqs
            )
        )
    return test_seqs
