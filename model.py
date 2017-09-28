import tensorflow as tf

def conv(x, w, b, stride, name, padding='SAME'):
    with tf.variable_scope('conv'):
        return tf.nn.conv2d(x,
                           filter=w,
                           strides=[1, stride, stride, 1],
                           padding=padding,
                           name=name) + b


def deconv(x, w, b, shape, stride, name):
    with tf.variable_scope('deconv'):
        return tf.nn.conv2d_transpose(x,
                                       filter=w,
                                       output_shape=shape,
                                       strides=[1, stride, stride, 1],
                                       padding='SAME',
                                       name=name) + b

def lrelu(x, alpha=0.2):
    with tf.variable_scope('leakyReLU'):
        return tf.maximum(x, alpha * x)

