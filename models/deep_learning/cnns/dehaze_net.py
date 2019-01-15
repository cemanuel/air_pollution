import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.framework import ops
import math
import os
import sys
import re
class DehazeNetModel(object):
    def __init__(self,
                    sess,
                    inputs,
                    labels,
                    stage_of_development,
                    num_of_classes,
                    min_bins=None,
                    max_bins=None): 
        self.labels = tf.reshape(tf.cast(labels, tf.float32), [-1, 1])
        self.stage_of_development = stage_of_development
        self.batch_inputs = tf.reshape(tf.cast(inputs, tf.float32), [-1, 224+15, 224+15, 3])
        self.filters_1 = tf.get_variable("DehazeNet/filters_1",
                                        dtype=tf.float32,
                                        initializer=tf.constant(np.load("/mnt/mnt/mounted_bucket/pretrained_weights/dehaze_net/conv1_0.npy"),shape=[5, 5, 3, 20]))
        self.num_of_classes = num_of_classes
        
        # FILTER # 1: ADD TO SUMMARY FOR VISUALIZATION IN TENSORBOARD - BEGINS 
        #filter1_pad= tf.zeros([5, 5, 3, 5], dtype=tf.float32)
        #filter1_b = tf.concat([self.filters_1, filter1_pad], 3)                                             #[5, 5, 3, 25]
        #filter1_c = tf.split(filter1_b, num_or_size_splits=25, axis=3)                                      #25 x [5, 5, 3, 1]
        #filter1_row0 = tf.concat(filter1_c[0:5], 0)                                                         #[25, 5, 3, 1]
        #filter1_row1 = tf.concat(filter1_c[5:10], 0)                                                        #[25, 5, 3, 1]
        #filter1_row2 = tf.concat(filter1_c[10:15], 0)                                                       #[25, 5, 3, 1]
        #filter1_row3 = tf.concat(filter1_c[15:20], 0)                                                       #[25, 5, 3, 1]
        #filter1_row4 = tf.concat(filter1_c[20:25], 0)                                                       #[25, 5, 3, 1]
        #filter1_d = tf.concat([filter1_row0, filter1_row1, filter1_row2, filter1_row3, filter1_row4], 1)    #[25, 25, 3, 1]
        #filter1_final_summary = tf.reshape(filter1_d, [1, 25, 25, 3])
        #tf.summary.image("Filter # 1: 25 - 5X5", filter1_final_summary)
        # FILTER # 1: ADD TO SUMMARY FOR VISUALIZATION IN TENSORBOARD - ENDS

        self.bias_1 = tf.get_variable("DehazeNet/bias_1",
                                    dtype=tf.float32,
                                    initializer=tf.constant(np.load("/mnt/mnt/mounted_bucket/pretrained_weights/dehaze_net/conv1_1.npy")))
        self.conv_1 = tf.nn.conv2d(self.batch_inputs, self.filters_1, [1, 1, 1, 1], padding="VALID") 

        self.output_of_relu_of_sum_of_conv_1_AND_bias_1 = tf.nn.relu(tf.nn.bias_add(self.conv_1, self.bias_1))

        print(self.output_of_relu_of_sum_of_conv_1_AND_bias_1)

        list_of_activations_conv_1 = tf.split(self.output_of_relu_of_sum_of_conv_1_AND_bias_1, num_or_size_splits=20, axis=3)

        for a_index, activation_ in enumerate(list_of_activations_conv_1):
            tf.summary.image("Activation for Conv 1" + str(a_index), tf.reshape(activation_, [-1, 224+11, 224+11, 1])) 

        def maxout(inputs, num_units, axis):
            inputs = ops.convert_to_tensor(inputs)
            shape = inputs.get_shape().as_list()
            num_channels = shape[axis]
            shape[axis] = -1
            shape += [num_channels // num_units]
            for i in range(len(shape)):
                if shape[i] is None:
                    shape[i] = gen_array_ops.shape(inputs)[i]
            outputs = math_ops.reduce_max(gen_array_ops.reshape(inputs, shape), -1, keep_dims=False)
            return outputs
        
        self.max_1 = maxout(tf.reshape(self.output_of_relu_of_sum_of_conv_1_AND_bias_1, [-1, 224+11, 224+11, 20]), 4, 3)
        self.max_1 = tf.reshape(self.max_1, [-1, 224+11, 224+11, 4])
        print(self.max_1)

        self.filters_2 = tf.get_variable("DehazeNet/filters_2",
                                    dtype=tf.float32,
                                    initializer=tf.constant(np.load("/mnt/mnt/mounted_bucket/pretrained_weights/dehaze_net/conv2-1x1_0.npy"), shape=[1, 1, 4, 16]))


        # FILTER # 2: ADD TO SUMMARY FOR VISUALIZATION IN TENSORBOARD - BEGINS 
        #filter2_c = tf.split(self.filters_2, num_or_size_splits=16, axis=3)   # 16 x [1, 1, 4, 1]
        #filter2_row0 = tf.concat(filter2_c[0:4], 0)   # [4, 1, 4, 1]
        #filter2_row1 = tf.concat(filter2_c[4:8], 0)   # [4, 1, 4, 1]
        #filter2_row2 = tf.concat(filter2_c[8:12], 0)  # [4, 1, 4, 1]
        #filter2_row3 = tf.concat(filter2_c[12:16], 0) # [4, 1, 4, 1]
        #filter2_d = tf.concat([filter2_row0, filter2_row1, filter2_row2, filter2_row3], 1) # [4, 4, 4, 1]
        #filter2_final_summary = tf.reshape(filter2_d, [1, 4, 4, 4])
        #tf.summary.image("Filter # 2: 16 - 1X1", filter2_final_summary)
        # FILTER # 2: ADD TO SUMMARY FOR VISUALIZATION IN TENSORBOARD - ENDS


        self.bias_2 = tf.get_variable("DehazeNet/bias_2",
                                dtype=tf.float32,
                                initializer=tf.constant(np.load("/mnt/mnt/mounted_bucket/pretrained_weights/dehaze_net/conv2-1x1_1.npy")))
        self.conv_2 = tf.nn.conv2d(self.max_1, self.filters_2, [1, 1, 1, 1], padding="VALID")
        self.output_of_relu_of_sum_of_conv_2_AND_bias_2 = tf.nn.relu(tf.nn.bias_add(self.conv_2, self.bias_2))
        print(self.output_of_relu_of_sum_of_conv_2_AND_bias_2)
        
        self.filters_3 = tf.get_variable("DehazeNet/filters_3",
                                        dtype=tf.float32,
                                        initializer=tf.constant(np.load("/mnt/mnt/mounted_bucket/pretrained_weights/dehaze_net/conv2-3x3_0.npy"), shape=[3, 3, 4, 16]))



        # FILTER # 3: ADD TO SUMMARY FOR VISUALIZATION IN TENSORBOARD - BEGINS
        #filter3_c = tf.split(self.filters_3, num_or_size_splits=16, axis=3)   # 16 x [1, 1, 4, 1]
        #filter3_row0 = tf.concat(filter3_c[0:4], 0)   # [12, 3, 4, 1]
        #filter3_row1 = tf.concat(filter3_c[4:8], 0)   # [12, 3, 4, 1]
        #filter3_row2 = tf.concat(filter3_c[8:12], 0)  # [12, 3, 4, 1]
        #filter3_row3 = tf.concat(filter3_c[12:16], 0) # [12, 3, 4, 1]
        #filter3_d = tf.concat([filter3_row0, filter3_row1, filter3_row2, filter3_row3], 1) # [12, 12, 4, 1]
        #filter3_final_summary = tf.reshape(filter3_d, [1, 12, 12, 4])
        #tf.summary.image("Filter # 3: 16 - 3X3", filter3_final_summary)
        # FILTER # 3: ADD TO SUMMARY FOR VISUALIZATION IN TENSORBOARD - ENDS


        self.bias_3 = tf.get_variable("DehazeNet/bias_3",
                                    dtype=tf.float32,
                                    initializer=tf.constant(np.load("/mnt/mnt/mounted_bucket/pretrained_weights/dehaze_net/conv2-3x3_1.npy")))
        self.conv_3 = tf.nn.conv2d(self.max_1, self.filters_3, [1, 1, 1, 1], padding="SAME")
        self.output_of_relu_of_sum_of_conv_3_AND_bias_3 = tf.nn.relu(tf.nn.bias_add(self.conv_3, self.bias_3))
        print(self.output_of_relu_of_sum_of_conv_3_AND_bias_3)
        
        self.filters_4 = tf.get_variable("DehazeNet/filters_4",
                                        dtype=tf.float32,
                                        initializer=tf.constant(np.load("/mnt/mnt/mounted_bucket/pretrained_weights/dehaze_net/conv2-5x5_0.npy"), shape=[5, 5, 4, 16]))


        # FILTER # 4: ADD TO SUMMARY FOR VISUALIZATION IN TENSORBOARD - BEGINS 
        #filter4_c = tf.split(self.filters_4, num_or_size_splits=16, axis=3)   # 16 x [5, 5, 4, 1]
        #filter4_row0 = tf.concat(filter4_c[0:4], 0)   # [20, 5, 4, 1]
        #filter4_row1 = tf.concat(filter4_c[4:8], 0)   # [20, 5, 4, 1]
        #filter4_row2 = tf.concat(filter4_c[8:12], 0)  # [20, 5, 4, 1]
        #filter4_row3 = tf.concat(filter4_c[12:16], 0) # [20, 5, 4, 1]
        #filter4_d = tf.concat([filter4_row0, filter4_row1, filter4_row2, filter4_row3], 1) # [20, 20, 4, 1]
        #filter4_final_summary = tf.reshape(filter4_d, [1, 20, 20, 4])
        #tf.summary.image("Filter # 4: 16 - 5X5", filter4_final_summary)
        # FILTER # 4: ADD TO SUMMARY FOR VISUALIZATION IN TENSORBOARD - ENDS


        self.bias_4 = tf.get_variable("DehazeNet/bias_4",
                                    dtype=tf.float32,
                                    initializer=tf.constant(np.load("/mnt/mnt/mounted_bucket/pretrained_weights/dehaze_net/conv2-5x5_1.npy")))
        self.conv_4 = tf.nn.conv2d(self.max_1, self.filters_4, [1, 1, 1, 1], padding="SAME")
        self.output_of_relu_of_sum_of_conv_4_AND_bias_4 = tf.nn.relu(tf.nn.bias_add(self.conv_4, self.bias_4)) 
        print(self.output_of_relu_of_sum_of_conv_4_AND_bias_4)


        self.filters_5 = tf.get_variable("DehazeNet/filters_5",
                                        dtype=tf.float32,
                                        initializer=tf.constant(np.load("/mnt/mnt/mounted_bucket/pretrained_weights/dehaze_net/conv2-7x7_0.npy"), shape=[7, 7, 4, 16]))


        # FILTER # 5: ADD TO SUMMARY FOR VISUALIZATION IN TENSORBOARD - BEGINS 
        #filter5_c = tf.split(self.filters_5, num_or_size_splits=16, axis=3)   # 16 x [7, 7, 4, 1]
        #filter5_row0 = tf.concat(filter5_c[0:4], 0)   # [28, 7, 4, 1]
        #filter5_row1 = tf.concat(filter5_c[4:8], 0)   # [28, 7, 4, 1]
        #filter5_row2 = tf.concat(filter5_c[8:12], 0)  # [28, 7, 4, 1]
        #filter5_row3 = tf.concat(filter5_c[12:16], 0) # [28, 7, 4, 1]
        #filter5_d = tf.concat([filter5_row0, filter5_row1, filter5_row2, filter5_row3], 1) # [28, 28, 4, 1]
        #filter5_final_summary = tf.reshape(filter5_d, [1, 28, 28, 4])
        #tf.summary.image("Filter # 5 - 16 - 7X7", filter5_final_summary)
        # FILTER # 5: ADD TO SUMMARY FOR VISUALIZATION IN TENSORBOARD - ENDS


        self.bias_5 = tf.get_variable("DehazeNet/bias_5",
                                    dtype=tf.float32,
                                    initializer=tf.constant(np.load("/mnt/mnt/mounted_bucket/pretrained_weights/dehaze_net/conv2-7x7_1.npy")))
        self.conv_5 =  tf.nn.conv2d(self.max_1, self.filters_5, [1, 1, 1, 1], padding="SAME")
       
        self.output_of_relu_of_sum_of_conv_5_AND_bias_5 = tf.nn.relu(tf.nn.bias_add(self.conv_5, self.bias_5))
        self.output_of_relu_of_sum_of_conv_5_AND_bias_5 = tf.reshape(self.output_of_relu_of_sum_of_conv_5_AND_bias_5, [-1, 224+11, 224+11, 16])


        #list_of_activations_conv_5 = tf.split(self.output_of_relu_of_sum_of_conv_5_AND_bias_5, num_or_size_splits=16, axis=3)

        #for a_index_conv_5, conv_5_activation_ in enumerate(list_of_activations_conv_5):
        #    tf.summary.image("Activation For Conv 5 " + str(a_index_conv_5), tf.reshape(conv_5_activation_, [-1, 224+11, 224+11, 1])) 
    

        self.concat_convs_with_16_filters = tf.concat([self.output_of_relu_of_sum_of_conv_2_AND_bias_2,
                                                        self.output_of_relu_of_sum_of_conv_3_AND_bias_3,
                                                        self.output_of_relu_of_sum_of_conv_4_AND_bias_4,
                                                        self.output_of_relu_of_sum_of_conv_5_AND_bias_5], axis=3)
        print(self.concat_convs_with_16_filters)
        self.max_pool_1 = tf.nn.max_pool(tf.nn.relu(self.concat_convs_with_16_filters), [1, 8, 8, 1], [1, 1, 1, 1], padding="VALID")
        print(self.max_pool_1)
        #self.final_conv_1 = tf.layers.conv2d(inputs=self.max_pool_1,
        #                            filters=1,
        #                            strides = (4, 4),
        #                            kernel_size=[8, 8],
        #                            padding="VALID",
        #                            activation=tf.nn.relu,
        #                            name="DehazeNet/final_conv_1")

        #self.max_pool_2 = tf.nn.max_pool(self.final_conv_1, [1, 8, 8, 1], [1, 8, 8, 1], padding='VALID')

        #self.final_conv_2 = tf.layers.conv2d(inputs=self.max_pool_2,
        #                                        filters=1,
        #                                        strides=(4, 4),
        #                                        kernel_size=[8, 8],
        #                                        padding="VALID",
        #                                        activation=tf.nn.relu,
        #                                        name="DehazeNet/final_conv_2")

        #print(self.final_conv_2)
        #self.logits = tf.reshape(self.final_conv_2, [-1, 1])

        self.filters_final = tf.get_variable("DehazeNet/filters_final",
                                            dtype=tf.float32,
                                            initializer=tf.constant(np.load("/mnt/mnt/mounted_bucket/pretrained_weights/dehaze_net/ip1_0.npy"), shape=[5, 5, 64, 1]))
        self.bias_final = tf.get_variable("DehazeNet/bias_final",
                                             dtype=tf.float32,
                                             initializer=tf.constant(np.load("/mnt/mnt/mounted_bucket/pretrained_weights/dehaze_net/ip1_1.npy")))
        self.conv_final = tf.nn.conv2d(self.max_pool_1, self.filters_final, [1, 1, 1, 1], padding="VALID")
        self.transmission = tf.reshape(tf.nn.relu(tf.nn.bias_add(self.conv_final, self.bias_final)), [-1, 224, 224, 1])
        self.average_pool = tf.layers.average_pooling2d(self.transmission, (4, 4), (4, 4), padding='VALID')
        self.average_pool_flat = tf.reshape(self.average_pool, [-1, 56*56])

        self.logits = tf.layers.dense(inputs=self.average_pool_flat, units=self.num_of_classes, activation=tf.nn.relu, name="DehazeNet/final_layer")

        self.logits = tf.reshape(tf.nn.relu(self.logits), [-1, num_of_classes])
        self.validation_logits = tf.reshape(tf.nn.relu(self.logits), [-1, num_of_classes])

        if self.num_of_classes == 1:
            self.predictions = self.logits
            self.validation_predictions = self.validation_logits
        else:
            self.predictions = tf.reshape(tf.cast(tf.argmax(self.logits, 1), tf.float32), [-1, 1])
            self.validation_predictions = tf.reshape(tf.cast(tf.argmax(self.validation_logits, 1), tf.float32), [-1, 1])


        self.all_variables = tf.trainable_variables()
        self.variables_trained_from_scratch = [x for x in self.all_variables if re.match(r'DehazeNet/final.*', x.name) != None] #tf.contrib.framework.get_variables('DehazeNet/final')
        #self.pre_trained_variables = [x for x in self.all_variables if re.match(r'DehazeNet/final.*', x.name) == None]
        print("DEHAZE NET - ALL VARIABLES: ", self.all_variables)
        print("DEHAZE NET - VARIABLES TRAINED FROM SCRATCH: ", self.variables_trained_from_scratch)