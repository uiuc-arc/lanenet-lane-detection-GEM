from unittest import TestCase

import tests.conf_tusimple
from lanenet.semantic_segmentation_zoo.vgg16_based_fcn import VGG16FCN

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class TestVGG16FCN(TestCase):
    def test_build_model(self):
        test_in_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input')
        model = VGG16FCN(phase='train', cfg=tests.conf_tusimple.lanenet_cfg)
        ret = model.build_model(test_in_tensor, name='vgg16fcn')
        for layer_name, layer_info in ret.items():
            print('layer name: {:s} shape: {}'.format(layer_name, layer_info['shape']))
