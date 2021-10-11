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


def conv1x3x3(mid_planes, stride=1):
    return ConvPadLayer(
        num_filters=mid_planes,
        kernel_size=(1, 3, 3),
        stride=(1, stride, stride),
        padding=(0, 1, 1),
        use_bias=False
    )


def conv3x1x1(planes, stride=1):
    return ConvPadLayer(
        num_filters=planes,
        kernel_size=(3, 1, 1),
        stride=(stride, 1, 1),
        padding=(1, 0, 0),
        use_bias=False
    )


def conv1x1x1(out_planes, stride=1):
    return ConvPadLayer(
        num_filters=out_planes,
        kernel_size=(1, 1, 1),
        stride=stride,
        padding=(0, 0, 0),
        use_bias=False
    )


class BasicBlock(keras_layers.Layer):
    expansion = 1

    def __init__(self,
                 in_planes,
                 planes,
                 stride=1,
                 downsample=None
                 ):
        super(BasicBlock, self).__init__()
        n_3d_parameters1 = in_planes * planes * 3 * 3 * 3
        n_2p1d_parameters1 = in_planes * 3 * 3 + 3 * planes
        mid_planes1 = n_3d_parameters1 // n_2p1d_parameters1
        self.conv1_s = conv1x3x3(mid_planes1, stride)
        self.bn1_s = keras_layers.BatchNormalization(axis=1)
        self.conv1_t = conv3x1x1(planes, stride)
        self.bn1_t = keras_layers.BatchNormalization(axis=1)

        n_3d_parameters2 = planes * planes * 3 * 3 * 3
        n_2p1d_parameters2 = planes * 3 * 3 + 3 * planes
        mid_planes2 = n_3d_parameters2 // n_2p1d_parameters2
        self.conv2_s = conv1x3x3(mid_planes2)
        self.bn2_s = keras_layers.BatchNormalization(axis=1)
        self.conv2_t = conv3x1x1(planes)
        self.bn2_t = keras_layers.BatchNormalization(axis=1)

        self.relu = keras_layers.ReLU()
        self.downsample = downsample
        self.stride = stride

    def call(self, x, **kwargs):
        residual = x

        out = self.conv1_s(x)
        out = self.bn1_s(out)
        out = self.relu(out)
        out = self.conv1_t(out)
        out = self.bn1_t(out)
        out = self.relu(out)

        out = self.conv2_s(out)
        out = self.bn2_s(out)
        out = self.relu(out)
        out = self.conv2_t(out)
        out = self.bn2_t(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(keras_layers.Layer):
    expansion = 4

    def __init__(self,
                 in_planes,
                 planes,
                 stride=1,
                 downsample=None
                 ):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1x1(planes)
        self.bn1 = keras_layers.BatchNormalization(axis=1)

        n_3d_parameters = planes * planes * 3 * 3 * 3
        n_2p1d_parameters = planes * 3 * 3 + 3 * planes
        mid_planes = n_3d_parameters // n_2p1d_parameters
        self.conv2_s = conv1x3x3(mid_planes, stride)
        self.bn2_s = keras_layers.BatchNormalization(axis=1)
        self.conv2_t = conv3x1x1(planes, stride)
        self.bn2_t = keras_layers.BatchNormalization(axis=1)

        self.conv3 = conv1x1x1(planes * self.expansion)
        self.bn3 = keras_layers.BatchNormalization(axis=1)
        self.relu = keras_layers.ReLU()
        self.downsample = downsample
        self.stride = stride

    def call(self, x, **kwargs):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2_s(out)
        out = self.bn2_s(out)
        out = self.relu(out)
        out = self.conv2_t(out)
        out = self.bn2_t(out)
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
                 n_classes=400):
        super(ResNet, self).__init__()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        n_3d_parameters = 3 * self.in_planes * conv1_t_size * 7 * 7
        n_2p1d_parameters = 3 * 7 * 7 + conv1_t_size * self.in_planes
        mid_planes = n_3d_parameters // n_2p1d_parameters

        self.conv1_s = ConvPadLayer(
            num_filters=mid_planes,
            kernel_size=(1, 7, 7),
            stride=(1, 2, 2),
            padding=(0, 3, 3),
            use_bias=False
        )
        self.bn1_s = keras_layers.BatchNormalization(axis=1)
        self.conv1_t = ConvPadLayer(
            num_filters=self.in_planes,
            kernel_size=(conv1_t_size, 1, 1),
            stride=(conv1_t_stride, 1, 1),
            padding=(conv1_t_size // 2, 0, 0),
            use_bias=False
        )
        self.bn1_t = keras_layers.BatchNormalization(axis=1)
        self.relu = keras_layers.ReLU()

        self.maxpool = MaxPoolPadLayer(
            pool_size=3,
            pool_stride=2,
            padding=(1, 1, 1)
        )
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.flatter = keras_layers.Flatten()
        self.avgpool = keras_layers.GlobalAveragePooling3D()
        self.fc = keras_layers.Dense(n_classes, use_bias=True)

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

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = keras_layers.Lambda(
                    lambda x: self._downsample_basic_block(x, planes * block.expansion, stride)
                )
            else:
                downsample = tf.keras.Sequential(
                    [
                        conv1x1x1(planes * block.expansion, stride),
                        keras_layers.BatchNormalization(axis=1)
                    ]
                )

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return tf.keras.Sequential(layers)

    def call(self, x):
        x = self.conv1_s(x)
        x = self.bn1_s(x)
        x = self.relu(x)
        x = self.conv1_t(x)
        x = self.bn1_t(x)
        x = self.relu(x)

        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = self.flatter(x)
        x = self.fc(x)

        return x


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
