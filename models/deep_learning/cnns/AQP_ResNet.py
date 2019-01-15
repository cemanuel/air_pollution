import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.framework import ops
import math
import os
import sys
import re
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
from tensorflow.contrib.slim.nets import resnet_v2

class AQPResNetModel(object):
    def __init__(self,
                    sess,
                    inputs,
                    labels,
                    stage_of_development,
                    model_path,
                    num_of_classes,
                    min_bins=None,
                    max_bins=None): 
        #self.resnet = tf.contrib.slim
        self.batch_inputs = tf.reshape(tf.cast(inputs, tf.float32), [-1, 224, 224, 3])
        self.stage_of_development = stage_of_development
        self.model_path = model_path
        self.labels = tf.reshape(tf.cast(labels, tf.float32), [-1, 1])
        self.num_of_classes = num_of_classes

        print(stage_of_development=="training")
        with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=0.0001)):
            self.logits, _ = resnet_v2.resnet_v2_50(self.batch_inputs, num_classes=self.num_of_classes, is_training=stage_of_development=="training", scope='resnet_v2_50')

        # Specify where the model checkpoint is (pretrained weights).
        self.model_path = model_path

        # Restore only the layers up to "logits" (included)
        # Calling function `init_fn(sess)` will load all the pretrained weights.
        self.variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['resnet_v2_50/logits'])
        print("ResNet 50 - VARIABLES TO RESTORE", self.variables_to_restore)
        self.init_fn = tf.contrib.framework.assign_from_checkpoint_fn(self.model_path, self.variables_to_restore)

        # Initialization operation from scratch for the new "logits" layers
        # `get_variables` will only return the variables whose name starts with the given pattern
        self.logits_variables = tf.contrib.framework.get_variables('resnet_v2_50/logits')
        self.logits_init = tf.variables_initializer(self.logits_variables)

        # ---------------------------------------------------------------------
        # Using tf.losses, any loss is added to the tf.GraphKeys.LOSSES collection
        # We can then call the total loss easily

        self.logits = tf.reshape(tf.nn.relu(self.logits), [-1, num_of_classes])
        self.validation_logits = tf.reshape(tf.nn.relu(self.logits), [-1, num_of_classes])

        if self.num_of_classes == 1:
            self.predictions = self.logits
            self.validation_predictions = self.validation_logits
        else:
            self.predictions = tf.reshape(tf.cast(tf.argmax(self.logits, 1), tf.float32), [-1, 1])
            self.validation_predictions = tf.reshape(tf.cast(tf.argmax(self.validation_predictions, 1), tf.float32), [-1, 1])

        self.all_variables = tf.trainable_variables()
        self.variables_trained_from_scratch = self.logits_variables 
        print("ResNet - ALL VARIABLES: ", self.all_variables)
        print("ResNet - VARIABLES TRAINED FROM SCRATCH: ", self.variables_trained_from_scratch)