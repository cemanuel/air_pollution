import tensorflow as tf
import numpy as np
import sys
sys.path.append('../models')
import math
from deep_learning.cnns.simple_cnn import SimpleCNNModel
from deep_learning.cnns.dehaze_net import DehazeNetModel
from deep_learning.cnns.AQP_vgg import AQPVGGModel
from deep_learning.cnns.AQP_ResNet import AQPResNetModel
from tensorflow.contrib.data import Dataset, Iterator
import tensorflow.contrib.slim as slim

IMG_HEIGHT = 240
IMG_WIDTH = 320

meta_data_names = ['filePath', 'webcamId', 'webcamLat', 'webcamLong', 'year', 'date', 'hour', 'range', 'pmValue']
dark_channel_names = ['Dark_Channel_' + str(i) for i in range(0, 100)]
atmospheric_light_names = ['Atmospheric_Light_' + str(i) for i in range(0, 3)]
transmission_names = ['Transmission_' + str(i) for i in range(0, 100)]
saturation_names = ['Saturation_1','Saturation_2','Saturation_3','Saturation_4','Saturation_5','Saturation_6','Saturation_7','Saturation_8','Saturation_9','Saturation_10']
contrast_names = ['Contrast_1']
power_spectrum_names = ['PS_1','PS_2','PS_3','PS_4','PS_5','PS_6','PS_7','PS_8','PS_9','PS_10']
weather_features = ['dir', 'spd', 'temp', 'dewpt', 'slp', 'rhx']
column_names_with_weather_features = meta_data_names + weather_features
column_names_with_haze_features =  meta_data_names + dark_channel_names + atmospheric_light_names + transmission_names + saturation_names + contrast_names + power_spectrum_names
all_column_names = meta_data_names + dark_channel_names + atmospheric_light_names + transmission_names + saturation_names + contrast_names + power_spectrum_names + weather_features

def parse_example(example_proto):
	feature_dict = {
		'filePath': tf.FixedLenFeature([], tf.float32),
		'image': tf.FixedLenFeature([], tf.string),
		'orig_height': tf.FixedLenFeature([], tf.int64),
		'orig_width': tf.FixedLenFeature([], tf.int64)}
	parsed_features = tf.parse_single_example(example_proto, features=feature_dict)
	pmValue = parsed_features['filePath']
	rgb_image = tf.decode_raw(parsed_features['image'], tf.float64)
	height = tf.cast(parsed_features['orig_height'], tf.int32)
	width = tf.cast(parsed_features['orig_width'], tf.int32)
	rgb_image = tf.reshape(rgb_image, [1, height, width, 3])
	rgb_image = tf.image.resize_image_with_crop_or_pad(rgb_image, target_height=480, target_width=480) * 255
	rgb_image = tf.reshape(tf.image.resize_area(rgb_image, [224+15, 224+15]), [224+15, 224+15, 3])
	return rgb_image, pmValue

def parse_example_vgg(example_proto):
	feature_dict = {
		'filePath': tf.FixedLenFeature([], tf.float32),
		'image': tf.FixedLenFeature([], tf.string),
		'orig_height': tf.FixedLenFeature([], tf.int64),
		'orig_width': tf.FixedLenFeature([], tf.int64)}
	parsed_features = tf.parse_single_example(example_proto, features=feature_dict)
	pmValue = parsed_features['filePath']
	rgb_image = tf.decode_raw(parsed_features['image'], tf.float64)
	height = tf.cast(parsed_features['orig_height'], tf.int32)
	width = tf.cast(parsed_features['orig_width'], tf.int32)
	rgb_image = tf.reshape(rgb_image, [1, height, width, 3])
	rgb_image = tf.image.resize_image_with_crop_or_pad(rgb_image, target_height=480, target_width=480) * 255
	rgb_image = tf.reshape(tf.image.resize_area(rgb_image, [224, 224]), [224, 224, 3])
	return rgb_image, pmValue

def parse_example_ResNet(example_proto):
	feature_dict = {
		'filePath': tf.FixedLenFeature([], tf.float32),
		'image': tf.FixedLenFeature([], tf.string),
		'orig_height': tf.FixedLenFeature([], tf.int64),
		'orig_width': tf.FixedLenFeature([], tf.int64)}
	parsed_features = tf.parse_single_example(example_proto, features=feature_dict)
	pmValue = parsed_features['filePath']
	rgb_image = tf.decode_raw(parsed_features['image'], tf.float64)
	height = tf.cast(parsed_features['orig_height'], tf.int32)
	width = tf.cast(parsed_features['orig_width'], tf.int32)
	rgb_image = tf.reshape(rgb_image, [1, height, width, 3])
	rgb_image = tf.image.resize_image_with_crop_or_pad(rgb_image, target_height=480, target_width=480) * 255
	rgb_image = tf.reshape(tf.image.resize_area(rgb_image, [224, 224]), [224, 224, 3])
	return rgb_image, pmValue

class AQPModel(object):
	def __init__(self,
					sess,
					batch_size,
					stage_of_development,
					learning_rate_decay_factor,
					type_of_model,
					summary_dir,
					experiment_folder, 
					type_of_optimizer,
					num_of_classes,
					total_num_of_training_examples,
					dropout=0.5,
					model_path=None,
					beta1=0.9,
					beta2=0.999,
					min_bins=None,
					max_bins=None,
					list_of_tfrecords_for_training=None,
					list_of_tfrecords_for_evaluation=None,
					training_with_eval=False,
					dict_of_filePath_to_num_of_examples_in_tfrecord=None):

		self.training_batch_size = 0
		self.batch_size = batch_size
		self.list_of_tr_datasets = []
		self.list_of_eval_datasets = []
		print("Training with dev", training_with_eval)
		print(sorted(list(dict_of_filePath_to_num_of_examples_in_tfrecord.keys())))

		if stage_of_development == "training":
			for tfrecord_for_training_example_ in list_of_tfrecords_for_training:
				current_tr_data = tf.contrib.data.TFRecordDataset(tfrecord_for_training_example_)
				if type_of_model == 'VGG':
					current_tr_data = current_tr_data.map(parse_example_vgg)
				elif type_of_model == 'ResNet':
					current_tr_data = current_tr_data.map(parse_example_ResNet)
				else:
					current_tr_data = current_tr_data.map(parse_example)
				current_tr_data = current_tr_data.shuffle(buffer_size=20000)
				current_tr_data = current_tr_data.repeat()
				current_tfrecord_batch_size = math.ceil(((float(dict_of_filePath_to_num_of_examples_in_tfrecord[tfrecord_for_training_example_]) * 1.0) / (float(total_num_of_training_examples) * 1.0)) * self.batch_size)
				self.training_batch_size += current_tfrecord_batch_size
				current_tr_data = current_tr_data.batch(current_tfrecord_batch_size)
				print(tfrecord_for_training_example_, dict_of_filePath_to_num_of_examples_in_tfrecord[tfrecord_for_training_example_], current_tfrecord_batch_size)
				self.list_of_tr_datasets.append(current_tr_data)

		if stage_of_development == "training":
			self.batch_size = self.training_batch_size

		self.single_eval_data = None
		if stage_of_development != "training":
			self.single_eval_data = tf.contrib.data.TFRecordDataset(list_of_tfrecords_for_evaluation)
			if type_of_model == 'VGG':
				self.single_eval_data = self.single_eval_data.map(parse_example_vgg)
			elif type_of_model == 'ResNet':
				self.single_eval_data = self.singe_eval_data.map(parse_example_ResNet)
			else:
				self.single_eval_data = self.single_eval_data.map(parse_example)
			self.single_eval_data = self.single_eval_data.shuffle(buffer_size=10000)
			self.single_eval_data = self.single_eval_data.repeat(1)
			self.single_eval_data = self.single_eval_data.batch(self.batch_size)


			#for tfrecord_for_evaluation_example_ in list_of_tfrecords_for_evaluation:
			#	current_eval_data = tf.contrib.data.TFRecordDataset(tfrecord_for_evaluation_example_)
			#	if type_of_model == 'VGG':
			#		current_eval_data = current_eval_data.map(parse_example_vgg)
			#	elif type_of_model == 'ResNet':
			#		current_eval_data = current_eval_data.map(parse_example_ResNet)
			#	else:
			#		current_eval_data = current_eval_data.map(parse_example)
			#	current_eval_data = current_eval_data.shuffle(buffer_size=10000)
			#	current_eval_data = current_eval_data.repeat(1)
			#	current_eval_data = current_eval_data.batch(self.batch_size)
			#	self.list_of_eval_datasets.append(current_eval_data)

		self.list_of_handles = []
		self.list_of_iterators = []
		self.list_of_batch_imgs = []
		self.list_of_batch_labels = []
		self.list_of_batch_imgs_and_batch_labels = []

		if stage_of_development == "training":
			for idx_ in range(len(list_of_tfrecords_for_training)):
				self.list_of_handles.append(tf.placeholder(tf.string, shape=[]))
				self.list_of_iterators.append(Iterator.from_string_handle(self.list_of_handles[idx_], self.list_of_tr_datasets[0].output_types, self.list_of_tr_datasets[0].output_shapes))
				batched_imgs, batched_labels = self.list_of_iterators[idx_].get_next()
				if type_of_model == 'DehazeNet':
					self.list_of_batch_imgs.append(tf.reshape(batched_imgs, [-1, 224+15, 224+15, 3]))
				else:
					self.list_of_batch_imgs.append(tf.reshape(batched_imgs, [-1, 224, 224, 3]))
				self.list_of_batch_labels.append(tf.reshape(batched_labels, [-1, 1]))
		else:
			self.single_eval_handle = tf.placeholder(tf.string, shape=[])
			self.single_eval_iterator = Iterator.from_string_handle(self.single_eval_handle, self.single_eval_data.output_types, self.single_eval_data.output_shapes)
			self.eval_batched_imgs, self.eval_batched_labels = self.single_eval_iterator.get_next()
			if type_of_model == 'DehazeNet':
				self.eval_batched_imgs = tf.reshape(self.eval_batched_imgs, [-1, 224+15, 224+15, 3])
			else:
				self.eval_batched_imgs = tf.reshape(self.eval_batched_imgs, [-1, 224, 224, 3])
			self.eval_batched_labels = tf.reshape(self.eval_batched_labels, [-1, 1])


			#for idx_ in range(len(list_of_tfrecords_for_evaluation)):
			#	self.list_of_handles.append(tf.placeholder(tf.string, shape=[]))
			#	self.list_of_iterators.append(Iterator.from_string_handle(self.list_of_handles[idx_], self.list_of_eval_datasets[0].output_types, self.list_of_eval_datasets[0].output_shapes))
			#	batched_imgs, batched_labels = self.list_of_iterators[idx_].get_next()
			#	if type_of_model == 'DehazeNet':
			#		self.list_of_batch_imgs.append(tf.reshape(batched_imgs, [-1, 224+15, 224+15, 3]))
			#	else:
			#		self.list_of_batch_imgs.append(tf.reshape(batched_imgs, [-1, 224, 224, 3]))
			#	self.list_of_batch_labels.append(tf.reshape(batched_labels, [-1, 1]))

		self.list_of_training_iterators = []
		self.single_eval_iterator = None

		if stage_of_development == "training":
			for tr_dataset_example_ in self.list_of_tr_datasets:
				validation_iterator = tr_dataset_example_.make_one_shot_iterator()
				self.list_of_training_iterators.append(validation_iterator)

		if stage_of_development == "evaluation":
			self.single_eval_iterator = self.single_eval_data.make_one_shot_iterator()

		self.row_indices = tf.placeholder(tf.int32, (self.batch_size,))
		self.row_indices_reshaped = tf.reshape(self.row_indices, [self.batch_size, 1])


		if stage_of_development == "training":
			self.batch_inputs = tf.gather_nd(tf.concat(self.list_of_batch_imgs, 0), self.row_indices_reshaped)
			self.batch_targets = tf.gather_nd(tf.concat(self.list_of_batch_labels, 0), self.row_indices_reshaped)
		else:
			self.batch_inputs = tf.gather_nd(self.eval_batch_imgs, self.row_indices_reshaped)
			self.batch_targets = tf.gather_nd(self.eval_batched_labels, self.row_indices_reshaped)

		self.stage_of_development = stage_of_development
		self.model_path = model_path
		self.type_of_optimizer = type_of_optimizer
		self.model = None
		self.beta1 = beta1
		self.beta2 = beta2
		self.pm_values = tf.gather_nd(tf.concat(self.list_of_batch_labels, 0), self.row_indices_reshaped)

		#if num_of_classes > 1:
		#	discrete_targets = tf.cast(self.batch_targets, dtype=tf.float32)
		#	discrete_targets = tf.reshape(discrete_targets, [-1, 1])
		#	min_bins = tf.reshape(tf.cast(min_bins, dtype=tf.float32), [1, -1])
		#	max_bins = tf.reshape(tf.cast(max_bins, dtype=tf.float32), [1, -1])
		#	c_1 =  tf.subtract(discrete_targets, min_bins)
		#	c_1 = tf.add(tf.cast(c_1 < 0, c_1.dtype) * 10000, tf.nn.relu(c_1))
		#	c_2 =  tf.subtract(discrete_targets * -1, max_bins)
		#	c_2 = tf.add(tf.cast(c_2 < 0, c_2.dtype) * 10000, tf.nn.relu(c_2))
		#	c = tf.add(c_1, c_2)
		#	self.batch_targets = tf.reshape(tf.argmin(c, 1), [-1, 1])

		self.is_training = tf.placeholder(tf.bool, shape=[])

		if type_of_model == 'DehazeNet':
			self.model = DehazeNetModel(sess,
										self.batch_inputs,
										self.batch_targets,
										self.stage_of_development,
										num_of_classes,
										min_bins=min_bins,
										max_bins=max_bins)
		elif type_of_model == "VGG":
			self.model = AQPVGGModel(sess,
									self.batch_inputs,
									self.batch_targets,
									self.stage_of_development,
									self.model_path,
									num_of_classes,
									self.is_training,
									min_bins=min_bins,
									max_bins=max_bins)
		elif type_of_model == "ResNet":
			self.model = AQPResNetModel(sess,
										self.batch_inputs,
										self.batch_targets,
										self.stage_of_development,
										self.model_path,
										num_of_classes,
										min_bins=min_bins,
										max_bins=max_bins)    
		else:
			self.model = SimpleCNNModel(sess, self.batch_inputs, self.batch_targets, self.stage_of_development)

		def return_predictions():
			return self.model.predictions
		def return_validation_predictions():
			return self.model.validation_predictions

		def return_MAE():
			return tf.reduce_mean(tf.abs(tf.subtract(self.model.predictions, self.model.labels)))
		def return_validation_MAE():
			return tf.reduce_mean(tf.abs(tf.subtract(self.model.validation_predictions, self.model.labels)))

		def return_MSE():
			return tf.reduce_mean(tf.square(tf.subtract(self.model.predictions, self.model.labels)))
		def return_validation_MSE():
			return tf.reduce_mean(tf.square(tf.subtract(self.model.validation_predictions, self.model.labels)))

		def return_MSLE():
			return tf.reduce_mean(tf.square(tf.subtract(tf.log(tf.add(self.model.predictions, 1.0)), tf.log(tf.add(self.model.labels, 1.0)))))
		def return_validation_MSLE():
			return tf.reduce_mean(tf.square(tf.subtract(tf.log(tf.add(self.model.validation_predictions, 1.0)), tf.log(tf.add(self.model.labels, 1.0)))))

		def return_R2_score():
			numerator = tf.reduce_sum(tf.square(tf.subtract(self.model.labels, self.model.predictions))) #Unexplained Error
			denominator = tf.reduce_sum(tf.square(tf.subtract(self.model.labels, tf.reduce_mean(self.model.labels)))) # Total Error
			return tf.subtract(1.0, tf.divide(numerator, denominator))
		def return_validation_R2_score():
			numerator = tf.reduce_sum(tf.square(tf.subtract(self.model.labels, self.model.validation_predictions))) #Unexplained Error
			denominator = tf.reduce_sum(tf.square(tf.subtract(self.model.labels, tf.reduce_mean(self.model.labels)))) # Total Error
			return tf.subtract(1.0, tf.divide(numerator, denominator))

		self.learning_rate = tf.placeholder(tf.float32, shape=[])
		self.partial_learning_rate = tf.placeholder(tf.float32, shape=[])
		self.global_step = tf.Variable(0, trainable=False)
		self.predictions  = tf.cond(self.is_training, return_predictions, return_validation_predictions)
		self.MAE_ = tf.cond(self.is_training, return_MAE, return_validation_MAE)
		self.R2_score_ = tf.cond(self.is_training, return_R2_score, return_validation_R2_score)
		self.MSE_ = tf.cond(self.is_training, return_MSE, return_validation_MSE)
		self.MSLE_ = tf.cond(self.is_training, return_MSLE, return_validation_MSLE)
		
		if self.stage_of_development == "training":
			self.global_eval_step = tf.Variable(0, trainable=False)
			self.global_eval_update_step_variable = tf.assign(self.global_eval_step, self.global_eval_step+1)
			tf.summary.scalar('MAE', self.MAE_)
			tf.summary.scalar('MSE', self.MSE_)
			tf.summary.scalar('MSLE', self.MSLE_)
			tf.summary.scalar('R2 Coefficient', self.R2_score_)

		if self.stage_of_development == "training" or self.stage_of_development == "resume_training":
			if type_of_model == 'DehazeNet' or type_of_model == 'VGG':
				partial_opt = None
				if self.type_of_optimizer == 'adam':
					partial_opt = tf.train.AdamOptimizer(learning_rate=self.partial_learning_rate, beta1=self.beta1, beta2=self.beta2)
				else:
					partial_opt = tf.train.GradientDescentOptimizer(learning_rate=self.partial_learning_rate)
				partial_gradient = tf.gradients(self.MAE_, self.model.variables_trained_from_scratch)
				self.partial_train_op = partial_opt.apply_gradients(zip(partial_gradient, self.model.variables_trained_from_scratch), global_step=self.global_step)

				if self.type_of_optimizer == 'adam':
					full_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1, beta2=self.beta2)
				else:
					full_opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
				full_gradient = tf.gradients(self.MAE_, self.model.all_variables)
				self.train_op = full_opt.apply_gradients(zip(full_gradient, self.model.all_variables), global_step=self.global_step)
			elif type_of_model == 'ResNet':
				partial_opt = tf.train.AdamOptimizer(learning_rate=self.partial_learning_rate, beta1=self.beta1, beta2=self.beta2)
				self.partial_train_op = slim.learning.create_train_op(self.MAE_, partial_opt, global_step=self.global_step, variables_to_train=self.model.variables_trained_from_scratch)

				full_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1, beta2=self.beta2)
				self.train_op = slim.learning.create_train_op(self.MAE_, full_opt, global_step=self.global_step, variables_to_train=self.model.all_variables)
			else:
				if self.type_of_optimizer == 'adam':
					opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1, beta2=self.beta2)
				else:
					opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
				gradient = tf.gradients(self.MAE_, self.model.all_variables)
				self.train_op = opt.apply_gradients(zip(gradient, self.model.all_variables), global_step=self.global_step)

		self.merged = tf.summary.merge_all()
		self.train_writer = tf.summary.FileWriter(summary_dir +  '/' + experiment_folder + '/train', sess.graph)
		self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=4)