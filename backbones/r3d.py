import tensorflow as tf
import tensorflow.keras.layers as keras_layers
from functools import partial


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(
        num_filters,
        stride=1
):
    return keras_layers.Conv3D(
        filters=num_filters,
        kernel_size=3,
        strides=stride,
        use_bias=False
    )


def conv1x1x1(
        num_filters,
        stride=1
):
    return keras_layers.Conv3D(
        filters=num_filters,
        kernel_size=1,
        strides=stride,
        use_bias=False
    )


class BasicBlock(tf.keras.keras_layers.Layer):
    expansion = 1

    def __init__(self, num_filters, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(num_filters, stride)
        self.bn1 = keras_layers.BatchNormalization(
            axis=1,
            fused=True
        )
        self.relu = keras_layers.ReLU()
        self.conv2 = conv3x3x3(num_filters)
        self.bn2 = keras_layers.BatchNormalization(axis=1, fused=True)
        self.downsample = downsample
        self.stride = stride

    def call(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(tf.keras.keras_layers.Layer):
    expansion = 4

    def __init__(self,
                 num_filters,
                 stride=1,
                 downsample=None
                 ):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1x1(num_filters)
        self.bn1 = keras_layers.BatchNormalization(axis=1, fused=True)
        self.conv2 = conv3x3x3(num_filters, stride)
        self.bn2 = keras_layers.BatchNormalization(axis=1, fused=True)
        self.conv3 = conv1x1x1(num_filters * self.expansion)
        self.bn3 = keras_layers.BatchNormalization(axis=1, fused=True)
        self.relu = keras_layers.ReLU()
        self.downsample = downsample
        self.stride = stride

    def call(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(tf.keras.Model):
    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400
                 ):
        super(ResNet, self).__init__()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.conv1 = layers.Conv3D(
            filters=self.in_planes,
            kernel_size=(conv1_t_size, 7, 7),
            strides=(conv1_t_stride, 2, 2),
            padding='same',
            use_bias=False
        )
        self.bn1 = layers.BatchNormalization(axis=1, fused=True)
        self.relu = layers.ReLU()
        self.maxpool = layers.MaxPool3D(
            kernel_size=3,
            strides=2,
            padding='same'
        )
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1],
                                       shortcut_type, stride=2)
        self.layer1 = self._make_layer(block, block_inplanes[2], layers[2],
                                       shortcut_type, stride=2)
        self.layer1 = self._make_layer(block, block_inplanes[3], layers[3],
                                       shortcut_type, stride=2)

        self.avgpool = keras_layers.AveragePooling3D(

        )

        pass

    def _downsample_basic_block(self, x, num_filters, stride):
        out = tf.nn.avg_pool3d(
            x,
            ksize=1,
            strides=stride,
            padding='VALID',
            data_format='NCDHW'
        )

        out_shape = tf.shape(out)

        zero_pads = tf.zeros(shape=(out_shape[0], num_filters - out_shape[1],
                                    out_shape[2],
                                    out_shape[3],
                                    out_shape[4]),
                             name='zero_pad'
                             )
        out = tf.concat(
            [
                out, zero_pads
            ],
            axis=1
        )
        return out

    def _make_layer(self, block, num_filters, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != num_filters * 4:
            if shortcut_type == 'A':
                downsample = partial(
                    self._downsample_basic_block,
                    num_filters=num_filters * 4,
                    stride=stride
                )
            else:
                downsample = tf.keras.Sequential(
                    [
                        conv1x1x1(num_filters * 4, stride),
                        keras_layers.BatchNormalization(axis=1, fused=True)
                    ]
                )

        layers = list()
        layers.append(
            block(
                num_filters=num_filters,
                stride=stride,
                downsample=downsample
            )
        )
        self.in_planes = num_filters * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    num_filters=num_filters
                )
            )

        return tf.keras.Sequential(*layers)

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
