from functools import partial

import tensorflow as tf
import tensorflow.keras.layers as keras_layers


def get_inplanes():
    return [64, 128, 256, 512]


class ConvPadLayer(tf.keras.layers.Layer):
    def __init__(self, num_filters, kernel_size,
                 stride,
                 padding,
                 use_bias):
        super(ConvPadLayer, self).__init__()
        self.padding = padding
        self.conv = keras_layers.Conv3D(
            filters=num_filters,
            kernel_size=kernel_size,
            strides=stride,
            use_bias=use_bias
        )

    def call(self, x):
        x = tf.pad(
            x,
            tf.constant(
                [
                    [0, 0],
                    [0, 0],
                    [self.padding[0], self.padding[0]],
                    [self.padding[1], self.padding[1]],
                    [self.padding[2], self.padding[2]]
                ]
            )
        )

        x = self.conv(x)

        return x


class MaxPoolPadLayer(tf.keras.layers.Layer):
    def __init__(self,
                 pool_size,
                 pool_stride,
                 padding
                 ):
        super(MaxPoolPadLayer, self).__init__()
        self.padding = padding
        self.pool = keras_layers.MaxPool3D(
            pool_size=pool_size,
            strides=pool_stride
        )

    def call(self, x):
        x = tf.pad(
            x,
            tf.constant(
                [
                    [0, 0],
                    [0, 0],
                    [self.padding[0], self.padding[0]],
                    [self.padding[1], self.padding[1]],
                    [self.padding[2], self.padding[2]]
                ]
            )
        )

        x = self.pool(x)
        return x


class BasicBlock(keras_layers.Layer):
    expansion = 1

    def __init__(self, num_filters, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvPadLayer(num_filters=num_filters, stride=stride,
                                  padding=(0, 0, 0), use_bias=False,
                                  kernel_size=3)
        self.bn1 = keras_layers.BatchNormalization(
            axis=1,
            fused=None
        )
        self.relu = keras_layers.ReLU()
        self.conv2 = ConvPadLayer(num_filters=num_filters, stride=1,
                                  padding=(0, 0, 0), use_bias=False,
                                  kernel_size=3)
        self.bn2 = keras_layers.BatchNormalization(axis=1, fused=None)
        self.downsample = downsample
        self.stride = stride

    def call(self, x, training=None):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(keras_layers.Layer):
    expansion = 4

    def __init__(self,
                 num_filters,
                 stride=1,
                 downsample=None
                 ):
        super(Bottleneck, self).__init__()
        self.conv1 = ConvPadLayer(
            num_filters=num_filters,
            kernel_size=1,
            padding=(0, 0, 0),
            use_bias=False,
            stride=1
        )
        self.bn1 = keras_layers.BatchNormalization(axis=1, fused=None)
        self.conv2 = ConvPadLayer(num_filters=num_filters, kernel_size=3,
                                  stride=stride, use_bias=False,
                                  padding=(1, 1, 1))
        self.bn2 = keras_layers.BatchNormalization(axis=1, fused=None)
        self.conv3 = ConvPadLayer(num_filters=num_filters * self.expansion,
                                  stride=1,
                                  kernel_size=1,
                                  use_bias=False,
                                  padding=(0, 0, 0))
        self.bn3 = keras_layers.BatchNormalization(axis=1, fused=None)
        self.relu = keras_layers.ReLU()
        self.downsample = downsample
        self.stride = stride

    def call(self, x, training=None):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out, training=training)

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
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 output_layers=None,
                 n_classes=400
                 ):
        super(ResNet, self).__init__()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.conv1 = ConvPadLayer(
            num_filters=self.in_planes,
            kernel_size=(conv1_t_size, 7, 7),
            stride=(conv1_t_stride, 2, 2),
            use_bias=False,
            padding=(conv1_t_size // 2, 3, 3)
        )
        self.bn1 = keras_layers.BatchNormalization(axis=1, fused=None)
        self.relu = keras_layers.ReLU()
        self.maxpool = MaxPoolPadLayer(pool_size=3,
                                       pool_stride=2,
                                       padding=(1, 1, 1))

        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1],
                                       shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2],
                                       shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3],
                                       shortcut_type, stride=2)

        self.fc = None
        if n_classes is not None:
            self.avgpool = keras_layers.GlobalAveragePooling3D()
            self.flatten = keras_layers.Flatten()
            self.fc = keras_layers.Dense(units=n_classes)
        self.output_layers = output_layers

        pass

    def _downsample_basic_block(self, x, num_filters, stride):
        out = tf.nn.avg_pool3d(
            x,
            ksize=1,
            strides=stride,
            padding='same',
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
        if stride != 1 or self.in_planes != num_filters * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    self._downsample_basic_block,
                    num_filters=num_filters * block.expansion,
                    stride=stride
                )
            else:
                downsample = tf.keras.Sequential(
                    [
                        ConvPadLayer(num_filters=num_filters *
                                                 block.expansion,
                                     stride=stride,
                                     use_bias=False,
                                     kernel_size=1,
                                     padding=(0, 0, 0)),
                        keras_layers.BatchNormalization(axis=1, fused=None)
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
        return tf.keras.Sequential(layers)

    def call(self, x, training=None):
        if self.output_layers is None:
            x = self.conv1(x)
            x = self.bn1(x, training=training)
            x = self.relu(x)
            if not self.no_max_pool:
                x = self.maxpool(x)
            x = self.layer1(x, training=training)
            x = self.layer2(x, training=training)
            x = self.layer3(x, training=training)
            x = self.layer4(x, training=training)

            if self.fc is not None:
                x = self.avgpool(x)
                x = self.flatten(x)
                x = self.fc(x)
            return x
        elif self.output_layers == 'blocks':
            output_dict = dict()
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            if not self.no_max_pool:
                x = self.maxpool(x)
            x = self.layer1(x)
            output_dict['block1'] = x
            x = self.layer2(x)
            output_dict['block2'] = x
            x = self.layer3(x)
            output_dict['block3'] = x
            x = self.layer4(x)
            output_dict['block4'] = x
            if self.fc is not None:
                x = self.avgpool(x)
                x = self.flatten(x)
                x = self.fc(x)
                output_dict['logits'] = x
            return output_dict
        else:
            output_dict = dict()
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            output_dict['init_prepool'] = x
            if not self.no_max_pool:
                x = self.maxpool(x)
                output_dict['init_postpool'] = x
            x = self.layer1(x)
            output_dict['block1'] = x
            x = self.layer2(x)
            output_dict['block2'] = x
            x = self.layer3(x)
            output_dict['block3'] = x
            x = self.layer4(x)
            output_dict['block4'] = x
            if self.fc is not None:
                x = self.avgpool(x)
                x = self.flatten(x)
                x = self.fc(x)
                output_dict['logits'] = x
            return output_dict


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model
