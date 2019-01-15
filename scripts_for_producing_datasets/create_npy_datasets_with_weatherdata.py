
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
import pandas as pd
from os.path import isfile
from datetime import datetime
from functools import reduce
import matplotlib.dates as mdates
import random
os.environ['CUDA_VISIBLE_DEVICES'] = ''

WEBCAMIDS = ['20265', '13342', '20258', '18879', '17603', '21587', '21673', '1066', '2045', '93', '5207', '6629', '1333', '30', '204', '22014']
PMRANGES = ["00-09", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69"]


def create_filtered_npy_with_weather_data_for_AQP(webcamId_, pmRange_):
	weatherdata=pd.read_csv("/mnt/mnt/mounted_bucket/weatherdata_PM_value_Temp_RHx_Spd_Slp_Dewpt_Dir_hour.csv")
	weatherdata.columns
	weatherdata = weatherdata.loc[:, ~weatherdata.columns.str.contains('^Unnamed')]
	train_dataset_by_webcamId_and_pmRange = np.load('/mnt/mnt/mounted_bucket/npy_datasets_by_webcamId_and_pmRange_filtered_cv2/TRAIN_raw_dataset_with_filePath_AND_webcamId_AND_webcamLat_AND_webcamLong_AND_year_AND_date_AND_hour_AND_range_AND_pm_ALLRANGES_webcamId_' + str(webcamId_) + '_pmRange_' + str(pmRange_) + '.npy')
	dev_dataset_by_webcamId_and_pmRange = np.load('/mnt/mnt/mounted_bucket/npy_datasets_by_webcamId_and_pmRange_filtered_cv2/DEV_raw_dataset_with_filePath_AND_webcamId_AND_webcamLat_AND_webcamLong_AND_year_AND_date_AND_hour_AND_range_AND_pm_ALLRANGES_webcamId_' + str(webcamId_) + '_pmRange_' + str(pmRange_) + '.npy')
	test_dataset_by_webcamId_and_pmRange = np.load('/mnt/mnt/mounted_bucket/npy_datasets_by_webcamId_and_pmRange_filtered_cv2/TEST_raw_dataset_with_filePath_AND_webcamId_AND_webcamLat_AND_webcamLong_AND_year_AND_date_AND_hour_AND_range_AND_pm_ALLRANGES_webcamId_' + str(webcamId_) + '_pmRange_' + str(pmRange_) + '.npy')
	meta_data_names = ['filePath', 'webcamId', 'webcamLat', 'webcamLong', 'year', 'date', 'hour', 'range', 'pm']
	column_names = meta_data_names
	
	df_train = pd.DataFrame(train_dataset_by_webcamId_and_pmRange, columns=column_names)
	df_train = df_train[df_train.filePath != 'filePath']
	df_train = df_train.drop_duplicates(subset="filePath", keep="first")
	df_train['hour'] = df_train['hour'].str.strip('.jpg')
	timestring = df_train['date'].astype(str)+df_train['hour'].astype(str)
	timestring = pd.to_datetime(timestring, format='%Y%m%d%H%M%S')
	df_train['datetime'] = pd.to_datetime(timestring, format='%Y%m%d%H%M')
	df_train['datetime'] = df_train['datetime'].astype(str)
	print(len(df_train.index))
	df_train_with_weatherdata = pd.merge(df_train, weatherdata,  how='left', left_on=['datetime'], right_on=['datetime'])
	print(len(df_train_with_weatherdata.index))
	df_train_with_weatherdata = df_train_with_weatherdata.dropna()
	print(len(df_train_with_weatherdata.index))
	df_train_with_weatherdata = df_train_with_weatherdata.drop(columns=['image_name', 'webcam_id', 'USAF', 'PM_value', 'datetime', 'hour_y'])
	print(df_train_with_weatherdata.head(1))
	npy_train_with_weatherdata = df_train_with_weatherdata.as_matrix()

	np.save('/mnt/mnt/mounted_bucket/npy_datasets_by_webcamId_and_pmRange_filtered_with_weather/TRAIN_with_filePath_AND_webcamId_AND_webcamLat_AND_webcamLong_AND_year_hour_AND_pm_AND_dir_AND_spd_AND_temp_AND_dewpt_AND_slp_AND_rhx_' + str(webcamId_) + '_pmRange_' + str(pmRange_), npy_train_with_weatherdata)

	df_dev = pd.DataFrame(dev_dataset_by_webcamId_and_pmRange, columns=column_names)
	df_dev = df_dev[df_dev.filePath != 'filePath']
	df_dev = df_dev.drop_duplicates(subset="filePath", keep="first")
	df_dev['hour'] = df_dev['hour'].str.strip('.jpg')
	timestring = df_dev['date'].astype(str)+df_dev['hour'].astype(str)
	timestring = pd.to_datetime(timestring, format='%Y%m%d%H%M%S')
	df_dev['datetime'] = pd.to_datetime(timestring, format='%Y%m%d%H%M')
	df_dev['datetime'] = df_dev['datetime'].astype(str)
	print(len(df_dev.index))
	df_dev_with_weatherdata = pd.merge(df_dev, weatherdata,  how='left', left_on=['datetime'], right_on=['datetime'])
	print(len(df_dev_with_weatherdata.index))
	df_dev_with_weatherdata = df_dev_with_weatherdata.dropna()
	print(len(df_dev_with_weatherdata.index))
	df_dev_with_weatherdata = df_dev_with_weatherdata.drop(columns=['image_name', 'webcam_id', 'USAF', 'PM_value', 'datetime', 'hour_y'])
	print(df_dev_with_weatherdata.head(1))
	npy_dev_with_weatherdata = df_dev_with_weatherdata.as_matrix()

	np.save('/mnt/mnt/mounted_bucket/npy_datasets_by_webcamId_and_pmRange_filtered_with_weather/DEV_with_filePath_AND_webcamId_AND_webcamLat_AND_webcamLong_AND_year_hour_AND_pm_AND_dir_AND_spd_AND_temp_AND_dewpt_AND_slp_AND_rhx_' + str(webcamId_) + '_pmRange_' + str(pmRange_), npy_dev_with_weatherdata)

	df_test = pd.DataFrame(test_dataset_by_webcamId_and_pmRange, columns=column_names)
	df_test = df_test[df_test.filePath != 'filePath']
	df_test = df_test.drop_duplicates(subset="filePath", keep="first")
	df_test['hour'] = df_test['hour'].str.strip('.jpg')
	timestring = df_test['date'].astype(str)+df_test['hour'].astype(str)
	timestring = pd.to_datetime(timestring, format='%Y%m%d%H%M%S')
	df_test['datetime'] = pd.to_datetime(timestring, format='%Y%m%d%H%M')
	df_test['datetime'] = df_test['datetime'].astype(str)
	print(len(df_test.index))
	df_test_with_weatherdata = pd.merge(df_test, weatherdata,  how='left', left_on=['datetime'], right_on=['datetime'])
	print(len(df_test_with_weatherdata.index))
	df_test_with_weatherdata = df_test_with_weatherdata.dropna()
	print(len(df_test_with_weatherdata.index))
	df_test_with_weatherdata = df_test_with_weatherdata.drop(columns=['image_name', 'webcam_id', 'USAF', 'PM_value', 'datetime', 'hour_y'])
	print(df_test_with_weatherdata.head(1))
	npy_test_with_weatherdata = df_test_with_weatherdata.as_matrix()

	np.save('/mnt/mnt/mounted_bucket/npy_datasets_by_webcamId_and_pmRange_filtered_with_weather/TEST_with_filePath_AND_webcamId_AND_webcamLat_AND_webcamLong_AND_year_hour_AND_pm_AND_dir_AND_spd_AND_temp_AND_dewpt_AND_slp_AND_rhx_' + str(webcamId_) + '_pmRange_' + str(pmRange_), npy_test_with_weatherdata)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--webcamId', type=str)
	parser.add_argument('--pmRange', type=str)
	args = parser.parse_args()
	create_filtered_npy_with_weather_data_for_AQP(args.webcamId, args.pmRange)

	