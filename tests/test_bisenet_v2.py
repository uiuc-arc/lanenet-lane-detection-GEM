from unittest import TestCase

import tests.conf_tusimple
from lanenet.semantic_segmentation_zoo.bisenet_v2 import BiseNetV2

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class TestBiseNetV2(TestCase):
    def test_build_model(self):
        test_in_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input')
        model = BiseNetV2(phase='train', cfg=tests.conf_tusimple.lanenet_cfg)
        ret = model.build_model(test_in_tensor, name='bisenetv2')
        for layer_name, layer_info in ret.items():
            print('layer name: {:s} shape: {}'.format(layer_name, layer_info['shape']))
