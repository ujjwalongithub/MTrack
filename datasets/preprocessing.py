import tensorflow as tf


def flip_boxes_horizontally(boxes):
    """Left-right flip the boxes.
     Args:
       boxes: Float32 tensor containing the bounding boxes -> [..., 4].
              Boxes are in normalized form meaning their coordinates vary
              between [0, 1].
              Each last dimension is in the form of [ymin, xmin, ymax, xmax].
     Returns:
       Flipped boxes.
     """
    ymin, xmin, ymax, xmax = tf.split(
        value=boxes, num_or_size_splits=4, axis=-1)
    flipped_xmin = tf.subtract(1.0, xmax)
    flipped_xmax = tf.subtract(1.0, xmin)
    flipped_boxes = tf.concat(
        [ymin, flipped_xmin, ymax, flipped_xmax], axis=-1)
    return flipped_boxes


def flip_image_horizontal_with_boxes(image: tf.Tensor,
                                     boxes: tf.Tensor
                                     ):
    image = tf.image.flip_left_right(image)
    boxes = flip_boxes_horizontally(boxes)
    return image, boxes


def random_horizontal_flip(image: tf.Tensor,
                           boxes: tf.Tensor
                           ):
    flip_probability = tf.random.uniform(
        shape=(), minval=0.0, maxval=1.0, dtype=tf.float32)

    image, boxes = tf.cond(
        tf.greater(flip_probability,
                   tf.constant(value=0.5, dtype=tf.float32)
                   ),
        flip_image_horizontal_with_boxes(image, boxes),
        lambda: image, boxes
    )

    return image, boxes


def rgb2grayscale(image):
    return tf.image.rgb_to_grayscale(image)


def random_rgb2gray(image, boxes):
    gray_probability = tf.random.uniform(shape=(), minval=0.0, maxval=1.0,
                                         dtype=tf.float32)
    image = tf.cond(
        tf.greater(
            gray_probability,
            tf.constant(0.5, dtype=tf.float32)
        ),
        lambda: rgb2grayscale(image),
        lambda: image
    )

    return image, boxes
