import tensorflow as tf
import tensorflow.keras.layers as layers


def conv3x3x3(
        num_filters,
        stride=1
):
    return layers.Conv3D(
        filters=num_filters,
        kernel_size=3,
        strides=stride,
        use_bias=False
    )


def conv1x1x1(
        num_filters,
        stride=1
):
    return layers.Conv3D(
        filters=num_filters,
        kernel_size=1,
        strides=stride,
        use_bias=False
    )


class BasicBlock(tf.Module):
    def __init__(self, num_filters, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(num_filters, stride)
        self.bn1 = layers.BatchNormalization(
            axis=1,
            fused=True
        )
        self.relu = layers.ReLU()
        self.conv2 = conv3x3x3(num_filters)
        self.bn2 = layers.BatchNormalization(axis=1, fused=True)
        self.downsample = downsample
        self.stride = stride

    def __call__(self, x, training=False):
        residual = x

        out = self.conv1(x, training=training)
        out = self.bn1(out, training=training)
        out = self.relu(out, training=training)
        out = self.conv2(out, training=training)
        out = self.bn2(out, training=training)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
