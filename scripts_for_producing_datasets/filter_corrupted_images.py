from PIL import Image
import numpy as np
import skimage.io as io
import tensorflow as tf
import os
import tensorflow as tf
from scipy.misc import imread
import argparse
import os
import sys
import math
import skimage
import skimage.transform
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = ''

WEBCAMIDS = ['20265', '13342', '1066', '2045', '93', '5207', '6629', '1333', '30', '204', '22014']
PMRANGES = ["00-09", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69"]
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def create_filtered_npy_for_AQP(webcamId, pmRange_, mounted_root_directory, directory_to_save_dataset_by_webcamId_and_pmRange, directory_to_save_filtered_dataset_by_webcamId_and_pmRange):
	dataset_by_webcamId_and_pmRange = np.load(mounted_root_directory + '/' + directory_to_save_dataset_by_webcamId_and_pmRange + '/raw_dataset_with_filePath_AND_webcamId_AND_webcamLat_AND_webcamLong_AND_year_AND_date_AND_hour_AND_range_AND_pm_webcamId_' + str(webcamId_) + '_pmRange_' + str(pmRange_) + '.npy')
	list_of_dataset = list(map(list, dataset_by_webcamId_and_pmRange))
	if len(list_of_dataset) >= 10:
		shuffled_indices = np.arange(len(dataset_by_webcamId_and_pmRange))
		np.random.shuffle(shuffled_indices)
		shuffled_original_dataset = dataset_by_webcamId_and_pmRange[shuffled_indices]
		num_of_examples_in_train_set = math.ceil(len(dataset_by_webcamId_and_pmRange) * 0.60)
		num_of_examples_in_dev_set = math.ceil(math.floor(len(dataset_by_webcamId_and_pmRange) * 0.20))
		num_of_examples_in_testing_set = math.floor(math.floor(len(dataset_by_webcamId_and_pmRange) * 0.2))
		train_dataset_examples = shuffled_original_dataset[:num_of_examples_in_train_set]
		dev_dataset_examples = shuffled_original_dataset[num_of_examples_in_train_set:num_of_examples_in_dev_set + num_of_examples_in_train_set]
		test_dataset_examples = shuffled_original_dataset[num_of_examples_in_dev_set + num_of_examples_in_train_set:]
		list_of_train_dataset_examples = list(map(list, train_dataset_examples))
		list_of_dev_dataset_examples = list(map(list, dev_dataset_examples))
		list_of_test_dataset_examples = list(map(list, test_dataset_examples))
		list_of_filtered_train_dataset_examples = []
		list_of_filtered_dev_dataset_examples = []
		list_of_filtered_test_dataset_examples = []
		for entry_train_example_ in list_of_train_dataset_examples:
			try:
				if os.path.getsize(entry_train_example_[0]) >= 10000:
					img_npy = imread(entry_train_example_[0])
					if img_npy.shape[2] == 3 or img_npy.shape[2] == 4:
						img_npy = skimage.transform.resize(img_npy, (500, 500, 3))
						list_of_filtered_train_dataset_examples.append(entry_train_example_)
						print(entry_train_example_[0])
			except:
				continue
		np.save(mounted_root_directory + '/' + directory_to_save_filtered_dataset_by_webcamId_and_pmRange + '/TRAIN_raw_dataset_with_filePath_AND_webcamId_AND_webcamLat_AND_webcamLong_AND_year_AND_date_AND_hour_AND_range_AND_pm_webcamId_' + str(webcamId_) + '_pmRange_' + str(pmRange_), np.array(list_of_filtered_train_dataset_examples).reshape(-1, 9))

		for entry_dev_example_ in list_of_dev_dataset_examples:
			try:
				if os.path.getsize(entry_dev_example_[0]) >= 10000:
					img_npy = imread(entry_dev_example_[0])
					if img_npy.shape[2] == 3 or img_npy.shape[2] == 4:
						img_npy = skimage.transform.resize(img_npy, (500, 500, 3))
						list_of_filtered_dev_dataset_examples.append(entry_dev_example_)
						print(entry_dev_example_[0])
			except:
				continue
		np.save(mounted_root_directory + '/' + directory_to_save_filtered_dataset_by_webcamId_and_pmRange + '/DEV_raw_dataset_with_filePath_AND_webcamId_AND_webcamLat_AND_webcamLong_AND_year_AND_date_AND_hour_AND_range_AND_pm_ALLRANGES_webcamId_' + str(webcamId_) + '_pmRange_' + str(pmRange_), np.array(list_of_filtered_dev_dataset_examples).reshape(-1, 9))

		for entry_test_example_ in list_of_test_dataset_examples:
			try:
				if os.path.getsize(entry_test_example_[0]) >= 10000:
					img_npy = imread(entry_test_example_[0])
					if img_npy.shape[2] == 3 or img_npy.shape[2] == 4:
						img_npy = skimage.transform.resize(img_npy, (500, 500, 3))
						list_of_filtered_test_dataset_examples.append(entry_test_example_)
						print(entry_test_example_[0])
			except:
				continue
		np.save(mounted_root_directory + '/' + directory_to_save_filtered_dataset_by_webcamId_and_pmRange + '/TEST_raw_dataset_with_filePath_AND_webcamId_AND_webcamLat_AND_webcamLong_AND_year_AND_date_AND_hour_AND_range_AND_pm_ALLRANGES_webcamId_' + str(webcamId_) + '_pmRange_' + str(pmRange_), np.array(list_of_filtered_test_dataset_examples).reshape(-1, 9))

def create_filtered_npy_for_AQP_evaluation(webcamId, pmRange_, mounted_root_directory, directory_to_save_dataset_by_webcamId_and_pmRange, directory_to_save_filtered_dataset_by_webcamId_and_pmRange):



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--webcamId', type=str)
	parser.add_argument('--pmRange', type=str)
	parser.add_argument('--type_of_evaluation', type=str)
	parser.add_argument('--mounted_root_directory', type=str)
	parser.add_argument('--directory_to_save_dataset_by_webcamId_and_pmRange', type=str)
	parser.add_argument('--directory_to_save_filtered_dataset_by_webcamId_and_pmRange', type=str)

	parser.add_argument()
	args = parser.parse_args()
	create_filtered_npy_for_AQP(args.webcamId,
								args.pmRange,
								args.mounted_root_directory,
								args.directory_to_save_dataset_by_webcamId_and_pmRange,
								args.directory_to_save_filtered_dataset_by_webcamId_and_pmRange)

