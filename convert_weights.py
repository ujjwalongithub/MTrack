import tensorflow as tf
from backbones.r3d import  generate_model
from pytorch_r3d import generate_model as gen
from loguru import logger
from torchsummary import summary
import torch
import numpy as np
import sys

logger.add('log.log')




if __name__ == "__main__":

    # Build TF2 model
    tf.keras.backend.set_image_data_format('channels_first')
    tf2_model = generate_model(152, n_classes=1039)
    tf2_model.build(input_shape=(None, 3, 16, 112, 112))
    tf_weights = tf2_model.weights
    for weight in tf_weights:
        logger.info('{} -> {}'.format(weight.name, weight.shape))



    logger.info(tf2_model.summary())
    # new_model = generate_model(152, n_classes=1039)
    # new_model.load_weights('backbones_pretrained/r3d_152_KM')
    #new_model = tf.keras.models.load_model('backbones_pretrained/r3d_152_KM')
    # inp = tf.ones(shape=(1,3,16,112,112))
    # out = new_model(inp, False)
    # logger.info(out)
    pytorch_model = gen(152, n_classes=1039).cuda()
    # pytorch_model.eval()
    state_dict = torch.load('./kenshohara_models/r3d152_KM_200ep.pth')['state_dict']
    logger.info(state_dict)
    for k in state_dict.keys():
        logger.info(k)
    pytorch_model.load_state_dict(
        state_dict
    )
    # inp = torch.ones([1,3,16,112,112]).cuda()
    # out = pytorch_model(inp)
    # logger.info(out.cpu().detach().numpy())
    pytorch_parameter_count = 0
    new_weights = list()
    #logger.info(tf2_model.trainable_weights)
    for name, param in pytorch_model.named_parameters():
        #if param.requires_grad:
        logger.info('{}->{}'.format(name, param.data.shape))
        logger.warning('{}->{}'.format(name, tf_weights[
            pytorch_parameter_count].name))
        continue
        tf_weight_shape = tf_weights[pytorch_parameter_count].shape
        if len(tf_weight_shape) == len(param.data.shape) == 1:
            tf2_model.trainable_weights[pytorch_parameter_count].assign(
                param.cpu().detach().numpy())
        elif len(tf_weight_shape) == len(param.data.shape) == 2:
            tf2_model.trainable_weights[pytorch_parameter_count].assign(
                np.transpose(param.cpu().detach().numpy()))
        else:
            tf2_model.trainable_weights[pytorch_parameter_count].assign(
                np.transpose(param.cpu().detach().numpy(),
                                            (2,3,4,1,0)))
        pytorch_parameter_count+=1

    logger.info('Total number of PyTorch weights = {}'.format(
        pytorch_parameter_count))
    logger.info('Total number of TF2 weights = {}'.format(len(tf_weights)))


    #logger.info(tf2_model.trainable_weights)

    sys.exit(-1)

    tf2_model.compute_output_shape(input_shape=(None,3,16,112,112))
    tf2_model.build(input_shape=(None, 3, 16, 112, 112))
    tf2_model.save('backbones_pretrained/r3d_152_KM')
    sample_input = np.random.randn(1,3,16,112,112)
    tf2_out = tf2_model(sample_input)
    pytorch_out = pytorch_model(sample_input).cpu().detach().numpy()
    logger.info(tf2_out)
    logger.info(pytorch_out)
    np.allclose(pytorch_out, tf2_out)






