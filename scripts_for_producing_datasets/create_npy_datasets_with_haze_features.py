import skimage
from skimage import transform
from skimage import io
from scipy.misc import imread
import imageio
import numpy as np
import cv2
import argparse
import pandas as pd
#import matplotlib.pyplot as plt
#import os
from sklearn.decomposition import PCA
from scipy import fftpack


NUM_COLUMNS_FOR_FEATURE_MATRIX = 233


WEBCAMIDS = ['20265', '13342', '20258', '18879', '17603', '21587', '21673', '1066', '2045', '93', '5207', '6629', '1333', '30', '204', '22014']
PMRANGES = ["00-09", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69"]


def radial_profile(img):
    Y,X,Z =img.shape
    y,x,z = np.indices((img.shape))
    r = np.sqrt((x -X/2)**2 + (y - Y/2)**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), img.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

def extract_power_spectrum(img, plot=False):
    F1 = fftpack.fft2(img)
    F2 = fftpack.fftshift( F1 )
    psd2D = 20*np.log(np.abs( F2 )).astype(np.uint8)
    features=radial_profile(psd2D)
    x=features.shape
    mod=np.mod(x,10)[0]
    feature=features[:-mod].reshape(10,-1)
    feature=np.sum(feature,axis=1)
    return feature

def saturation(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[1],None,[10],[0,256])
    features = hist
    return features

def contrast(img):
    #first we calculate luminance
    YCB = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y=YCB[:,:,0]
    N,P=Y.shape
    C=np.sqrt(np.abs(N*P*np.sum(Y**2)-np.sum(Y)**2))/N*P
    return C

def get_dark_channel(I, w):
	"""Get the dark channel prior in the (RGB) image data.
	Parameters
	-----------
	I:  an M * N * 3 numpy array containing data ([0, L-1]) in the image where
		M is the height, N is the width, 3 represents R/G/B channels.
	w:  window size
	Return
	-----------
	An M * N array for the dark channel prior ([0, L-1]).
	"""
	M, N, _ = I.shape
	padded = np.pad(I, ((int(w / 2), int(w / 2)), (int(w / 2), int(w / 2)), (0, 0)), 'edge')
	darkch = np.zeros((M, N))
	for i, j in np.ndindex(darkch.shape):
		darkch[i, j] = np.min(padded[i:i + w, j:j + w, :])  # CVPR09, eq.5
	return darkch

def get_transmission(I, A, darkch, omega, w):
	"""Get the transmission esitmate in the (RGB) image data.
	Parameters
	-----------
	I:       the M * N * 3 RGB image data ([0, L-1]) as numpy array
	A:       a 3-element array containing atmosphere light
			([0, L-1]) for each channel
	darkch:  the dark channel prior of the image as an M * N numpy array
	omega:   bias for the estimate
	w:       window size for the estimate
	Return
	-----------
	An M * N array containing the transmission rate ([0.0, 1.0])
	"""
	return 1 - omega * get_dark_channel(I / A, w)  # CVPR09, eq.12

def get_atmosphere(I, darkch, p):
	"""Get the atmosphere light in the (RGB) image data.
	Parameters
	-----------
	I:      the M * N * 3 RGB image data ([0, L-1]) as numpy array
v	p:      percentage of pixels for estimating the atmosphere light
	Return
	-----------
	A 3-element array containing atmosphere light ([0, L-1]) for each channel
	"""
	# reference CVPR09, 4.4
	M, N = darkch.shape
	flatI = I.reshape(M * N, 3)
	flatdark = darkch.ravel()
	searchidx = (-flatdark).argsort()[:int(M * N * p)]  # find top M * N * p indexes
	# return the highest intensity for each channel
	return np.max(flatI.take(searchidx, axis=0), axis=0)

def save_output_image(output_ndarray, path_to_output_image):
	np.save(path_to_output_image, output_ndarray)
	#skimage.io.imsave(path_to_output_image, output_ndarray)

def extract_dark_channel_features(img_npy):
	darkchannel = get_dark_channel(img_npy, 15)
	feature_matrix = np.zeros((10, 10))
	for i, j in np.ndindex(feature_matrix.shape):
		feature_matrix[i, j] = np.median(darkchannel[i:(i+1)*50, j:(j+1)*50])
	return feature_matrix.ravel(), darkchannel

def extract_atmosphere_features(img_npy, darkchannel):
	A_ = get_atmosphere(img_npy, darkchannel, 0.01)
	return A_.ravel(), A_

def extract_transmission_features(img_npy, A_, darkchannel):
	transmission_matrix = get_transmission(img_npy, A_, darkchannel, 0.95, 15)
	feature_matrix = np.zeros((10, 10))
	for i, j in np.ndindex(feature_matrix.shape):
		feature_matrix[i, j] = np.median(transmission_matrix[i:(i+1)*50, j:(j+1)*50])
	return feature_matrix.ravel(), transmission_matrix

def extract_features_from_image(path_to_input_image, webcamId_, webcamLat_, webcamLong_, year_, date_, hour_, range_, pm_):
	img_npy = imread(path_to_input_image)
	img_npy = skimage.transform.resize(img_npy, (500, 500, 3))
	img_npy = np.array(img_npy).astype(np.float64)
	dark_channel_features, darkchannel = extract_dark_channel_features(img_npy)
	atmosphere_features, A_ = extract_atmosphere_features(img_npy, darkchannel)
	transmission_features, transmission_matrix = extract_transmission_features(img_npy, A_, darkchannel)

	img = cv2.imread(path_to_input_image)
	power_spectrum=extract_power_spectrum(img)
	saturation_histogram=saturation(img)
	C=contrast(img)

	return np.concatenate((np.array([path_to_input_image, webcamId_, webcamLat_, webcamLong_, year_, date_, hour_, range_, pm_]).reshape(1, -1),
							np.array(dark_channel_features.astype(np.float32)).reshape(1, -1),
							np.array(atmosphere_features.astype(np.float32)).reshape(1, -1),
							np.array(transmission_features.astype(np.float32)).reshape(1, -1),
							np.array(saturation_histogram.astype(np.float32)).reshape(1, -1),
							np.array(power_spectrum.astype(np.float32)).reshape(1, -1),
							np.array(C.astype(np.float32)).reshape(1, -1)), axis=1)

def extract_features_by_webcamId_and_pmRange_(webcamId_, pmRange_, mounted_root_directory):
	train_dataset_by_webcamId_and_pmRange_ = np.load(mounted_root_directory + '/npy_datasets_by_webcamId_and_pmRange_filtered_cv2/TRAIN_raw_dataset_with_filePath_AND_webcamId_AND_webcamLat_AND_webcamLong_AND_year_AND_date_AND_hour_AND_range_AND_pm_ALLRANGES_webcamId_' + str(webcamId_) + '_pmRange_' + str(pmRange_) + '.npy')
	dev_dataset_by_webcamId_and_pmRange_ = np.load(mounted_root_directory + '/npy_datasets_by_webcamId_and_pmRange_filtered_cv2/DEV_raw_dataset_with_filePath_AND_webcamId_AND_webcamLat_AND_webcamLong_AND_year_AND_date_AND_hour_AND_range_AND_pm_ALLRANGES_webcamId_' + str(webcamId_) + '_pmRange_' + str(pmRange_) + '.npy')
	test_dataset_by_webcamId_and_pmRange_ = np.load(mounted_root_directory + '/npy_datasets_by_webcamId_and_pmRange_filtered_cv2/TEST_raw_dataset_with_filePath_AND_webcamId_AND_webcamLat_AND_webcamLong_AND_year_AND_date_AND_hour_AND_range_AND_pm_ALLRANGES_webcamId_' + str(webcamId_) + '_pmRange_' + str(pmRange_) + '.npy')
	TRAIN_file_paths, TRAIN_webcamIds, TRAIN_webcamLats, TRAIN_webcamLongs, TRAIN_years, TRAIN_dates, TRAIN_hours, TRAIN_pmRanges, TRAIN_pmValues = map(list, zip(*tuple(train_dataset_by_webcamId_and_pmRange_)))
	DEV_file_paths, DEV_webcamIds, DEV_webcamLats, DEV_webcamLongs, DEV_years, DEV_dates, DEV_hours, DEV_pmRanges, DEV_pmValues = map(list, zip(*tuple(dev_dataset_by_webcamId_and_pmRange_)))
	TEST_file_paths, TEST_webcamIds, TEST_webcamLats, TEST_webcamLongs, TEST_years, TEST_dates, TEST_hours, TEST_pmRanges, TEST_pmValues = map(list, zip(*tuple(test_dataset_by_webcamId_and_pmRange_)))
	print("NUM OF TRAINING EXAMPLES: ", len(TRAIN_file_paths))
	print("NUM OF DEV EXAMPLES: ", len(DEV_file_paths))
	print("NUM OF TEST EXAMPLES: ", len(TEST_file_paths))
	for train_file_path_, train_webcamId_, train_webcamLat_, train_webcamLong_, train_year_, train_date_, train_hour_, train_pmRange_, train_pmValue_ in zip(TRAIN_file_paths, TRAIN_webcamIds, TRAIN_webcamLats, TRAIN_webcamLongs, TRAIN_years, TRAIN_dates, TRAIN_hours, TRAIN_pmRanges, TRAIN_pmValues):
		try:
			train_current_feature_matrix = extract_features_from_image(train_file_path_, train_webcamId_, train_webcamLat_, train_webcamLong_, train_year_, train_date_, train_hour_, train_pmRange_, train_pmValue_)
			train_full_matrix = np.load(TRAIN_PATH_feature_matrix_prefix + "_webcamId_" + webcamId_ + "_pmRange_" + pmRange_ + ".npy")
			train_updated_matrix = np.concatenate((train_full_matrix.reshape(-1, NUM_COLUMNS_FOR_FEATURE_MATRIX), np.array(train_current_feature_matrix).reshape(1, NUM_COLUMNS_FOR_FEATURE_MATRIX)), axis=0)
			print("*****************************************************************************")
			print("TRAIN")
			print(train_updated_matrix.shape)
			print(train_current_feature_matrix.shape)
			print("*****************************************************************************")
			np.save(TRAIN_PATH_feature_matrix_prefix + "_webcamId_" + webcamId_ + "_pmRange_" + pmRange_, train_updated_matrix)
		except:
			continue

	for dev_file_path_, dev_webcamId_, dev_webcamLat_, dev_webcamLong_, dev_year_, dev_date_, dev_hour_, dev_pmRange_, dev_pmValue_ in zip(DEV_file_paths, DEV_webcamIds, DEV_webcamLats, DEV_webcamLongs, DEV_years, DEV_dates, DEV_hours, DEV_pmRanges, DEV_pmValues):
		try:
			dev_current_feature_matrix = extract_features_from_image(dev_file_path_, dev_webcamId_, dev_webcamLat_, dev_webcamLong_, dev_year_, dev_date_, dev_hour_, dev_pmRange_, dev_pmValue_)
			dev_full_matrix = np.load(DEV_PATH_feature_matrix_prefix + "_webcamId_" + webcamId_ + "_pmRange_" + pmRange_ + ".npy")
			dev_updated_matrix = np.concatenate((dev_full_matrix.reshape(-1, NUM_COLUMNS_FOR_FEATURE_MATRIX), np.array(dev_current_feature_matrix).reshape(1, NUM_COLUMNS_FOR_FEATURE_MATRIX)), axis=0)
			print("*****************************************************************************")
			print("DEV")
			print(dev_updated_matrix.shape)
			print(dev_current_feature_matrix.shape)
			print("*****************************************************************************")
			np.save(DEV_PATH_feature_matrix_prefix + "_webcamId_" + webcamId_ + "_pmRange_" + pmRange_, dev_updated_matrix)
		except:
			continue

	for test_file_path_, test_webcamId_, test_webcamLat_, test_webcamLong_, test_year_, test_date_, test_hour_, test_pmRange_, test_pmValue_ in zip(TEST_file_paths, TEST_webcamIds, TEST_webcamLats, TEST_webcamLongs, TEST_years, TEST_dates, TEST_hours, TEST_pmRanges, TEST_pmValues):
		try:
			test_current_feature_matrix = extract_features_from_image(test_file_path_, test_webcamId_, test_webcamLat_, test_webcamLong_, test_year_, test_date_, test_hour_, test_pmRange_, test_pmValue_)
			test_full_matrix = np.load(TEST_PATH_feature_matrix_prefix + "_webcamId_" + webcamId_ + "_pmRange_" + pmRange_ + ".npy")
			test_updated_matrix = np.concatenate((test_full_matrix.reshape(-1, NUM_COLUMNS_FOR_FEATURE_MATRIX), np.array(test_current_feature_matrix).reshape(1, NUM_COLUMNS_FOR_FEATURE_MATRIX)), axis=0)
			print("*****************************************************************************")
			print("TEST")
			print(test_updated_matrix.shape)
			print(test_current_feature_matrix.shape)
			print("*****************************************************************************")
			np.save(TEST_PATH_feature_matrix_prefix + "_webcamId_" + webcamId_ + "_pmRange_" + pmRange_, test_updated_matrix)
		except:
			continue
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--webcamId', type=str)
	parser.add_argument('--pmRange', type=str)
	parser.add_argument('--mounted_root_directory', type=str)

	args = parser.parse_args()
	meta_data_names = ['filePath', 'webcamId', 'webcamLat', 'webcamLong', 'year', 'date', 'hour', 'range', 'pm']
	dark_channel_names = ['Dark_Channel_' + str(i) for i in range(0, 100)]
	atmospheric_light_names = ['Atmospheric_Light_' + str(i) for i in range(0, 3)]
	transmission_names = ['Transmission_' + str(i) for i in range(0, 100)]
	saturation_names = ['Saturation 1','Saturation 2','Saturation 3','Saturation 4','Saturation 5','Saturation 6','Saturation 7','Saturation 8','Saturation 9','Saturation 10']
	power_spectrum_names = ['PS 1','PS 2','PS 3','PS 4','PS 5','PS 6','PS 7','PS 8','PS 9','PS 10']
	contrast_names = ['Contrast 1']
	TRAIN_PATH_feature_matrix_prefix = mounted_root_directory + '/npy_datasets_by_webcamId_and_pmRange_haze_features/TRAIN_darkChannel_AND_atmosphere_AND_transmission_AND_saturation_AND_powerSpectrum_contrast'
	DEV_PATH_feature_matrix_prefix = mounted_root_directory + '/npy_datasets_by_webcamId_and_pmRange_haze_features/DEV_darkChannel_AND_atmosphere_AND_transmission_AND_saturation_AND_powerSpectrum_contrast'
	TEST_PATH_feature_matrix_prefix = mounted_root_directory + '/npy_datasets_by_webcamId_and_pmRange_haze_features/TEST_darkChannel_AND_atmosphere_AND_transmission_AND_saturation_AND_powerSpectrum_contrast'
	np.save(TRAIN_PATH_feature_matrix_prefix + "_webcamId_" + args.webcamId + "_pmRange_" + args.pmRange, meta_data_names + dark_channel_names + atmospheric_light_names + transmission_names + saturation_names + contrast_names + power_spectrum_names)
	np.save(DEV_PATH_feature_matrix_prefix + "_webcamId_" + args.webcamId + "_pmRange_" + args.pmRange, meta_data_names + dark_channel_names + atmospheric_light_names + transmission_names + saturation_names + contrast_names + power_spectrum_names)
	np.save(TEST_PATH_feature_matrix_prefix + "_webcamId_" + args.webcamId + "_pmRange_" + args.pmRange, meta_data_names + dark_channel_names + atmospheric_light_names + transmission_names + saturation_names + contrast_names + power_spectrum_names)
	extract_features_by_webcamId_and_pmRange_(args.webcamId, args.pmRange)


