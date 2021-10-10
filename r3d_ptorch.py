import sys
import math
from functools import partial
from backbones.r3d import generate_model
import pytorch_r3d
import numpy as np
import torch
from torchsummary import summary
from loguru import logger
import tensorflow as tf


logger.add('log.log')

NUM_BLOCKS = [3, 24, 36, 3]
BLOCK_TYPE = 'bottleneck'

BN_NUM = 0


def split_pytorch_name(name: str):
    if name[:5] == 'layer':
        splits = name.split('.')
        pass


def split_for_block(name: str):
    global BN_NUM
    tf_name = ''
    splits = name.split('.')
    layer_num = int(splits[0][5:])
    if layer_num == 1:
        factor = 0
    else:
        factor = sum(NUM_BLOCKS[:(layer_num-1)]) - 1
    block_num = int(splits[1])
    factor = factor + block_num + 1
    if factor == 0:
        tf_name = tf_name + '{}/'.format(BLOCK_TYPE)
    else:
        tf_name = tf_name + '{}_{}/'.format(BLOCK_TYPE, factor)

    layer_type = splits[2]
    if 'bn' in layer_type:


if __name__ == '__main__':
    tf.keras.backend.set_image_data_format('channels_first')
    model = pytorch_r3d.generate_model(152, n_classes=1039)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    numparams = sum([np.prod(p.size()) for p in model_parameters])
    inp = torch.randn((1, 3, 8, 112, 112)).cuda()
    loader = torch.load('./kenshohara_models/r3d152_KM_200ep.pth')
    state_dict = loader['state_dict']
    for k, v in state_dict.items():
        logger.info('{}->{}'.format(k, v.shape))

    model.load_state_dict(torch.load(
        './kenshohara_models/r3d152_KM_200ep.pth')['state_dict'])
    # sys.exit(-1)
    # model.cuda()
    #out = model(inp)
    # logger.info(out)
    '''
    summary(model, input_size=(3,16,112,112))
    print(numparams)
    for name, param in model.named_parameters():
        logger.info('{}->{}'.format(name, param.shape))

    '''
    tf_model = generate_model(152, n_classes=1039)
    tf_model.build(input_shape=(None, 3, 16, 112, 112))
    tf_trainable_weights = tf_model.trainable_weights
    for w in tf_trainable_weights:
        logger.info('{}->{}'.format(w.name, w.shape))

    sys.exit(-1)
    logger.warning('{} ----- {}'.format(len(state_dict),
                   len(tf_trainable_weights)))

    pytorch_names = list()
    pytorch_weights = list()
    for k, v in model.named_parameters():
        if not v.requires_grad:
            continue
        if 'num_batches_tracked' in k:
            continue
        pytorch_names.append(k)
        pytorch_weights.append(v)
    print(len(pytorch_weights))
    weights = list()
    tfweights = tf_model.get_weights()
    print(len(tfweights))

    for i in range(len(pytorch_weights)):
        tf_weight = tf_trainable_weights[i]
        tf_weight_shape = tf_weight.shape
        pytorch_weight_shape = pytorch_weights[i].shape
        logger.info(
            '{} ({}) -> {} ({})'.format(pytorch_names[i], pytorch_weight_shape, tf_weight.name, tf_weight_shape))
        if len(tf_weight_shape) == len(pytorch_weight_shape) == 1:
            weights.append(pytorch_weights[i].cpu().detach().numpy())
        elif len(tf_weight_shape) == len(pytorch_weight_shape) == 2:
            weights.append(np.transpose(
                pytorch_weights[i].cpu().detach().numpy()))
        else:
            weights.append(np.transpose(
                pytorch_weights[i].cpu().detach().numpy(), (2, 3, 4, 1, 0)))

        # print(tf_weight_shape)
        # print(pytorch_weight_shape)

    tf_model.set_weights(weights)
