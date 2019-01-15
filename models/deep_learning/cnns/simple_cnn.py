import tensorflow as tf
class SimpleCNNModel(object):
    def __init__(self, sess, inputs, labels, stage_of_development): 
        self.labels = tf.cast(labels, tf.float32)
        self.batch_inputs = tf.reshape(tf.cast(inputs, tf.float32), [-1, 240, 320, 3])
        self.stage_of_development = stage_of_development
        self.learning_rate = learning_rate 

        #[Batch Size X 240 X 320 X 1] -> [Batch Size X 240 X 320 X 4]
        conv1 = tf.layers.conv2d(inputs=self.batch_inputs, 
                                    filters=4,
                                    kernel_size=[5, 5],
                                    padding="same",
                                    activation=tf.nn.relu)

        list_of_activations_conv_1 = tf.split(conv1, num_or_size_splits=4, axis=3)

        for a_index, activation_ in enumerate(list_of_activations_conv_1):
            tf.summary.image("Activation for Conv 1 " + str(a_index), tf.reshape(activation_, [-1, 240, 320, 1])) 



        # FILTER # 1: ADD TO SUMMARY FOR VISUALIZATION IN TENSORBOARD - BEGINS 
        filter1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "conv2d/kernel")[0] #[5, 5, 3, 4]
        filter1_c = tf.split(filter1, num_or_size_splits=4, axis=3)   # 4 x [5, 5, 3, 1]
        filter1_row0 = tf.concat(filter1_c[0:2], 0)   # [10, 5, 3, 1]
        filter1_row1 = tf.concat(filter1_c[2:4], 0)   # [10, 5, 3, 1]
        filter1_d = tf.concat([filter1_row0, filter1_row1], 1) # [10, 10, 3, 1]
        filter1_final_summary = tf.reshape(filter1_d, [1, 10, 10, 3])
        tf.summary.image("Filter # 1: 4 - 5X5", filter1_final_summary)
        # FILTER # 1: ADD TO SUMMARY FOR VISUALIZATION IN TENSORBOARD - ENDS


        #[Batch Size X 240 X 320 X 4] -> [Batch Size X (240 - 2)/2 + 1 X (320 - 2)/2 + 1 X 4] -> [Batch Size X 120 X 160 X 4]
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        #[Batch Size X 120 X 160 X 4] -> [Batch Size X 120 X 160 X 8]
        conv2 = tf.layers.conv2d(inputs=pool1,
                                    filters=8,
                                    kernel_size=[5, 5],
                                    padding="same",
                                    activation=tf.nn.relu)


        # FILTER # 2: ADD TO SUMMARY FOR VISUALIZATION IN TENSORBOARD - BEGINS 
        filter2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "conv2d_1/kernel")[0] #[5, 5, 4, 8]
        filter2_pad= tf.zeros([5, 5, 4, 1], dtype=tf.float32)
        filter2_b = tf.concat([filter2, filter2_pad], 3)     #[5, 5, 4, 9]
        filter2_c = tf.split(filter2_b, num_or_size_splits=9, axis=3)                      # 9 x [5, 5, 4, 1]
        filter2_row0 = tf.concat(filter2_c[0:3], 0)    # [15, 5, 4, 1]
        filter2_row1 = tf.concat(filter2_c[3:6], 0)   # [15, 5, 4, 1]
        filter2_row2 = tf.concat(filter2_c[6:9], 0)  # [15, 5, 4, 1]
        filter2_d = tf.concat([filter2_row0, filter2_row1, filter2_row2], 1) # [15, 15, 4, 1]
        filter2_final_summary = tf.reshape(filter2_d, [1, 15, 15, 4])
        tf.summary.image("Filter #2: 8 - 5X5", filter2_final_summary)
        # FILTER # 2: ADD TO SUMMARY FOR VISUALIZATION IN TENSORBOARD - ENDS


        #[Batch Size X 120 X 160 X 8] -> [Batch Size X (120 - 2)/2 + 1 X (160 - 2)/2 + 1 X 8] -> [Batch Size X 60 X 80 X 8]
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        #[Batch Size X 60 X 80 X 8] -> [Batch Size X 60 X 80 X 16]
        conv3 = tf.layers.conv2d(inputs=pool2,
                                    filters=16,
                                    kernel_size=[5, 5],
                                    padding="same",
                                    activation=tf.nn.relu)
        #[Batch Size X 60 X 80 X 16] -> [Batch Size X (60-2)/2 + 1 X (80-2)/2 + 1 X 16] -> [Batch Size X 30 X 40 X 16]
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
        #[Batch Size X 30 X 40 X 16] -> [Batch Size X 30 X 40 X 32]
        conv4 = tf.layers.conv2d(inputs=pool3,
                                    filters=32,
                                    kernel_size=[5, 5],
                                    padding="same",
                                    activation=tf.nn.relu)

        list_of_activations_conv_4 = tf.split(conv4, num_or_size_splits=32, axis=3)

        for a_index, activation_ in enumerate(list_of_activations_conv_4):
            tf.summary.image("Activation for Conv 4 " + str(a_index), tf.reshape(activation_, [-1, 30, 40, 1])) 
        #[Batch Size X 30 X 40 X 32] -> [Batch Size X (30-2)/2 + 1 X (40-2)/2 + 1 X 32] -> [Batch Size X 15 X 20 X 32]
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
        pool4_flat = tf.reshape(pool4, [-1, 15 * 20 * 32])

        dense = tf.layers.dense(inputs=pool4_flat, units=9600, activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=self.stage_of_development=="training")
        dense2 = tf.layers.dense(inputs=dropout, units=1000, activation=tf.nn.relu)
        dropout2 = tf.layers.dropout(inputs=dense2, rate=0.4, training=self.stage_of_development=="training")
        self.logits = tf.layers.dense(inputs=dropout, units=1)

        self.all_variables = tf.trainable_variables()

    def return_predictions(self):
        return self.logits
    def return_MAE(self):
        return tf.reduce_mean(tf.abs(tf.subtract(self.logits,self.labels)))
    def return_MSE(self):
        return tf.reduce_mean(tf.square(tf.subtract(self.logits,self.labels)))
    def return_MSLE(self):
        return tf.reduce_mean(tf.square(tf.subtract(tf.log(tf.add(self.logits, 1.0)), tf.log(tf.add(self.labels, 1.0)))))
    def return_best_fit_slope(self):
        mean_predictions = tf.reduce_mean(self.logits)
        mean_labels = tf.reduce_mean(self.labels)
        mean_product_of_predictions_and_labels = tf.reduce_mean(tf.multiply(self.logits, self.labels))
        return tf.divide(tf.subtract(tf.multiply(mean_predictions, mean_labels), mean_product_of_predictions_and_labels), tf.square(mean_predictions))
    def return_explained_variance_score(self):
        mean_error = tf.reduce_mean(tf.subtract(self.labels, self.logits))
        numerator = tf.square(tf.reduce_mean(tf.subtract(self.labels, tf.subtract(self.logits, mean_error))))
        denominator = tf.square(tf.reduce_mean(tf.subtract(self.labels, tf.reduce_mean(self.labels))))
        return tf.subtract(1.0, tf.divide(numerator, denominator))
    def return_R2_score(self):
        numerator = tf.reduce_sum(tf.square(tf.subtract(self.labels, self.logits))) #Unexplained Error
        denominator = tf.reduce_sum(tf.square(tf.subtract(self.labels, tf.reduce_mean(self.labels)))) # Total Error
        return tf.subtract(1.0, tf.divide(numerator, denominator))
