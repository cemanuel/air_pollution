
import numpy as np
import os
import re
import math
from os.path import isfile
from pyspark.mllib.fpm import FPGrowth
from datetime import datetime
from functools import reduce
import matplotlib.dates as mdates
import random
import argparse
from pyspark.sql import SQLContext
from pyspark.sql.types import *
import pyspark
sc = pyspark.SparkContext()
sqlContext = SQLContext(sc)

WEBCAMIDS = ['20265', '13342', '20258', '18879', '17603', '21587', '21673', '1066', '2045', '93', '5207', '6629', '1333', '30', '204', '22014']

'''
RDD Format for Hourly Data
Column Data->   0: webcam_id 		||
				1: month			||
				2: date 			||
				3: hour 			||
				4: webcam_lat 		||
				5: webcam_long 		||
				6: AQ_site_id 		||
				7: AQ_monitor_id 	||
				8: AQ_lat 			||
				9: AQ_long			||
				10: PM_value		||
'''

def get_rid_of_first_line(lines_of_data):
	get_rid_of_first_line = lambda line: line != '"webcam_id","month","date","image_name","webcam_lat","webcam_long","AQ_site_id","AQ_monitor_id","AQ_lat","AQ_long","PM_value"'
	lines_of_data = lines_of_data.filter(get_rid_of_first_line)
	git_rid_of_nans = lambda line: 'NA' not in line.strip().split(',')
	lines_of_data = lines_of_data.filter(git_rid_of_nans)
	return lines_of_data

def get_pmValues(lines_of_data):
	reordered_data = lines_of_data.map(lambda entry: (entry[0][0], entry[1]))
	reordered_data_npy = reordered_data.collect()
	list_of_pmValues = list(map(lambda x: float(x[1]), reordered_data_npy))
	current_max_pmValues = max(list_of_pmValues)
	current_min_pmValues = min(list_of_pmValues)
	upper_bound = int(math.ceil(current_max_pmValues / 10.0)) * 10
	min_bins = []
	max_bins = []
	pmRanges = []
	for current_bound_ in range(0, upper_bound, 10):
		min_bins.append(float(current_bound_))
		max_bins.append(float(current_bound_ + 10))
		pmRanges.append(str(current_bound_).zfill(2) + "-" + str(current_bound_).zfill(2))
	return min_bins, max_bins, pmRanges

def get_webcamIds(lines_of_data):
	reordered_data = lines_of_data.map(lambda entry: (entry[0][0], entry[1]))
	reordered_data_npy = reordered_data.collect()
	list_of_webcamIds = list(map(lambda x: int(x[0]), reordered_data_npy))
	return webcamIds

def create_filtered_webcamId_AND_year_TO_LIST_OF_COUNTS_OF_rangesForPMconcentration(lines_of_hourly_data):
	lines_of_hourly_data = get_rid_of_first_line(lines_of_hourly_data)
	print(lines_of_hourly_data.first())
	create_webcamId_AND_year_AND_date_AND_hour_AND_webcamLat_AND_webcamLong_TO_pm = lambda line:((line.strip().split(',')[0].replace('"',''), re.match('(\d\d\d\d).*', line.strip().split(',')[2].replace('"','')).group(1), line.strip().split(',')[2].replace('"', ''), line.strip().split(',')[3].replace('"', ''), line.strip().split(',')[4], line.strip().split(',')[5]), float(line.strip().split(',')[10].replace('"', '')))
	webcamId_AND_year_AND_date_AND_hour_AND_webcamLat_AND_webcamLong_TO_pm = lines_of_hourly_data.map(create_webcamId_AND_year_AND_date_AND_hour_AND_webcamLat_AND_webcamLong_TO_pm)
	filter_by_webcamId = lambda entry: entry[0][0] in ['20265', '13342', '20258', '18879', '17603', '21587', '21673', '1066', '2045', '93', '5207', '6629', '1333', '30', '204', '22014']
	webcamId_AND_date_AND_hour_AND_webcamLat_AND_webcamLong_TO_avgPM_filtered_by_webcamId = webcamId_AND_year_AND_date_AND_hour_AND_webcamLat_AND_webcamLong_TO_pm.filter(filter_by_webcamId).groupByKey().mapValues(list).map(lambda entry: (entry[0], np.average(entry[1])))
	get_pmValues(webcamId_AND_date_AND_hour_AND_webcamLat_AND_webcamLong_TO_avgPM_filtered_by_webcamId)
	print(webcamId_AND_date_AND_hour_AND_webcamLat_AND_webcamLong_TO_avgPM_filtered_by_webcamId.count())

	list_of_collected_partitions = []

	for current_min_bin_, current_max_bin_, current_pmRange_ in zip(MIN_BINS, MAX_BINS, PMRANGES):
		print(current_min_bin_, current_max_bin_, current_pmRange_)
		current_partition = webcamId_AND_date_AND_hour_AND_webcamLat_AND_webcamLong_TO_avgPM_filtered_by_webcamId.filter(lambda entry: entry[1] >= float(current_min_bin_) and entry[1] <= float(current_max_bin_)).map(lambda entry: ((entry[0][0], entry[0][1], entry[0][2], entry[0][3], entry[0][4], entry[0][5], str(current_pmRange_)), entry[1]))
		current_partition_npy = current_partition.collect() 	
		print(len(current_partition_npy))																	             
		list_of_collected_partitions.append(current_partition_npy)
	
	return list_of_collected_partitions

def create_single_npy_dataset(lines_of_hourly_data, path_to_mounted_root_directory):
	list_of_groups_of_data = create_filtered_webcamId_AND_year_TO_LIST_OF_COUNTS_OF_rangesForPMconcentration(lines_of_hourly_data)
	list_of_groups_of_data_prep_for_concat = []
	total_sum = 0
	for pre_current_idx_ in range(len(list_of_groups_of_data)):
		print(len(list_of_groups_of_data[pre_current_idx_]))
		total_sum += len(list_of_groups_of_data[pre_current_idx_])
	print(total_sum)
	for current_idx_ in range(len(list_of_groups_of_data)):
		list_of_groups_of_data_prep_for_concat.append(np.array(list(map(lambda x: np.array([path_to_mounted_root_directory + "/" + x[0][0].zfill(8) + "/" + x[0][0].zfill(8) + "_" + x[0][2] + "_" + x[0][3], x[0][0], x[0][4], x[0][5], x[0][1], x[0][2], x[0][3], x[0][6], x[1]]).reshape(1, 9), list_of_groups_of_data[current_idx_]))).reshape(-1, 9))
	return np.concatenate(tuple(list_of_groups_of_data_prep_for_concat), axis=0)

def create_datasets_by_webcamId_pmRange(full_dataset, path_to_mounted_root_directory, directory_to_save_raw_dataset_by_webcamId_and_pmRange):
	lengths_ = []
	directory_to_store_datasets = path_to_mounted_root_directory  + '/' +  directory_to_save_raw_dataset_by_webcamId_and_pmRange
	if not os.path.exists(directory_to_store_datasets):
		os.makedirs(directory_to_store_datasets)

	pattern_to_save_dataset_by_webcamId_AND_pmRange_prefix = directory_to_store_datasets  + "/raw_dataset_with_filePath_AND_webcamId_AND_webcamLat_AND_webcamLong_AND_year_AND_date_AND_hour_AND_range_AND_pm"
	for webcamId_ in WEBCAMIDS:
		for pmRange_ in PMRANGES:
			filtered_list = list(filter(lambda x: x[1] == webcamId_ and x[7] == pmRange_, full_dataset))
			lengths_.append(len(filtered_list))
			path_to_save_filtered_filter = pattern_to_save_dataset_by_webcamId_AND_pmRange_prefix + "_webcamId_" + str(webcamId_) + "_pmRange_" + pmRange_
			np.save(path_to_save_filtered_filter, np.array(filtered_list).reshape(-1, 9))
	print(sum(lengths_))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--mounted_root_directory', type=str)
	parser.add_argument('--directory_to_save_dataset_by_webcamId_and_pmRange', type=str)
	parser.add_argument('--webcamIds_for_filtering', nargs=('*'))
	parser.add_argument('--pmRangeNames', nargs=('*'))
	parser.add_argument('--MIN_BINS', nargs=('*'))
	parser.add_argument('--MAX_BINS', nargs=('*'))
	args = parser.parse_args()
	directory_to_save_raw_dataset_by_webcamId_and_pmRange = args.directory_to_save_dataset_by_webcamId_and_pmRange
	path_to_mounted_root_directory = args.mounted_root_directory
	path_to_hourly_data_relative_to_root_mounted_directory = 'AMOS_AQ_hourly.csv'
	print(path_to_mounted_root_directory + "/" + path_to_hourly_data_relative_to_root_mounted_directory)
	lines_of_hourly_data = sc.textFile(path_to_mounted_root_directory + "/" + path_to_hourly_data_relative_to_root_mounted_directory)
	print(lines_of_hourly_data.first())
	res = create_single_npy_dataset(lines_of_hourly_data, path_to_mounted_root_directory)
	print(res.shape)
	create_datasets_by_webcamId_pmRange(res, path_to_mounted_root_directory)

