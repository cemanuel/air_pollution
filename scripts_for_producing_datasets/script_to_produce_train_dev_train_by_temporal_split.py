import skimage
from skimage import transform
import skimage.transform
import imageio
import numpy as np
import pandas as pd
import urllib
import numpy as np
import skimage.io as io
import tensorflow as tf
from scipy.misc import imread
import argparse
import os
import sys
import math
import skimage
import datetime

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

num_of_columns_for_haze_feature_matrix = len(column_names_with_haze_features)
num_of_columns_for_weather_feature_matrix = len(column_names_with_weather_features)


TRAIN_PATH_TO_WEATHER_NPY_BY_WEBCAM_ID_AND_PMRANGE_PREFIX = '/mnt/mnt/mounted_bucket/npy_datasets_by_webcamId_and_pmRange_filtered_with_weather/TRAIN_with_filePath_AND_webcamId_AND_webcamLat_AND_webcamLong_AND_year_hour_AND_pm_AND_dir_AND_spd_AND_temp_AND_dewpt_AND_slp_AND_rhx_'
DEV_PATH_TO_WEATHER_NPY_BY_WEBCAM_ID_AND_PMRANGE_PREFIX = '/mnt/mnt/mounted_bucket/npy_datasets_by_webcamId_and_pmRange_filtered_with_weather/DEV_with_filePath_AND_webcamId_AND_webcamLat_AND_webcamLong_AND_year_hour_AND_pm_AND_dir_AND_spd_AND_temp_AND_dewpt_AND_slp_AND_rhx_'
TEST_PATH_TO_WEATHER_NPY_BY_WEBCAM_ID_AND_PMRANGE_PREFIX = '/mnt/mnt/mounted_bucket/npy_datasets_by_webcamId_and_pmRange_filtered_with_weather/TEST_with_filePath_AND_webcamId_AND_webcamLat_AND_webcamLong_AND_year_hour_AND_pm_AND_dir_AND_spd_AND_temp_AND_dewpt_AND_slp_AND_rhx_'

TRAIN_PATH_TO_HAZE_FEATURES_NPY_BY_WEBCAM_ID_AND_PMRANGE_PREFIX = '/mnt/mnt/mounted_bucket/npy_datasets_by_webcamId_and_pmRange_haze_features/TRAIN_dark_channel_AND_atmosphere_AND_transmission_features_webcamId_'
DEV_PATH_TO_HAZE_FEATURES_NPY_BY_WEBCAM_ID_AND_PMRANGE_PREFIX = '/mnt/mnt/mounted_bucket/npy_datasets_by_webcamId_and_pmRange_haze_features/DEV_dark_channel_AND_atmosphere_AND_transmission_features_webcamId_'
TEST_PATH_TO_HAZE_FEATURES_NPY_BY_WEBCAM_ID_AND_PMRANGE_PREFIX = '/mnt/mnt/mounted_bucket/npy_datasets_by_webcamId_and_pmRange_haze_features/TEST_dark_channel_AND_atmosphere_AND_transmission_features_webcamId_'

PATH_TO_SAVE_DATASET_BY_WEBCAMID_OF_HAZE_AND_WEATHER_FEATURES_PREFIX = '/mnt/mnt/mounted_bucket/npy_datasets_by_webcamId_with_haze_features_AND_weather_features/dataset_of_haze_features_AND_weather_features_webcamId_'


PATH_TO_TRAIN_DATASET_OF_HAZE_AND_WEATHER_FEATURES_BY_WEBCAMID_YEAR_AND_PMRANGE_PREFIX = '/mnt/mnt/mounted_bucket/npy_datasets_by_webcamId_year_and_pmRange_with_haze_features_AND_weather_features/TRAIN_dataset_of_haze_and_weather_features_webcamId_'
PATH_TO_DEV_DATASET_OF_HAZE_AND_WEATHER_FEATURES_BY_WEBCAMID_PREFIX = '/mnt/mnt/mounted_bucket/npy_datasets_by_webcamId_year_and_pmRange_with_haze_features_AND_weather_features/DEV_dataset_of_haze_and_weather_features_webcamId_'
PATH_TO_TEST_DATASET_OF_HAZE_AND_WEATHER_FEATURES_BY_WEBCAMID_PREFIX = '/mnt/mnt/mounted_bucket/npy_datasets_by_webcamId_year_and_pmRange_with_haze_features_AND_weather_features/TEST_dataset_of_haze_and_weather_features_webcamId_'



def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def create_npy_of_haze_features_and_weather_features(webcamId_):
    entire_dataset_by_webcam_id_with_haze_features = None
    ISNONE_entire_dataset_by_webcam_id_with_haze_features = True
    entire_dataset_by_webcam_id_with_weather_features = None
    ISNONE_entire_dataset_by_webcam_id_with_feature_features = True
    pmRanges = ['00-09', '10-19', '20-29', '30-39', '40-49', '50-59']
    for current_pmRange_ in pmRanges:
        print(webcamId_, current_pmRange_)
        TRAIN_dataset_weather_by_webcamId_pmRange = np.load(TRAIN_PATH_TO_WEATHER_NPY_BY_WEBCAM_ID_AND_PMRANGE_PREFIX +  str(webcamId_) + '_pmRange_' + str(current_pmRange_) + '.npy').reshape(-1, num_of_columns_for_weather_feature_matrix)
        TRAIN_dataset_hazefeatures_by_webcamId_pmRange = np.load(TRAIN_PATH_TO_HAZE_FEATURES_NPY_BY_WEBCAM_ID_AND_PMRANGE_PREFIX  +  str(webcamId_) + '_pmRange_' + str(current_pmRange_) + '.npy')[1:].reshape(-1, num_of_columns_for_haze_feature_matrix)
        DEV_dataset_weather_by_webcamId_pmRange = np.load(DEV_PATH_TO_WEATHER_NPY_BY_WEBCAM_ID_AND_PMRANGE_PREFIX +  str(webcamId_) + '_pmRange_' + str(current_pmRange_) + '.npy').reshape(-1, num_of_columns_for_weather_feature_matrix)
        DEV_dataset_hazefeatures_by_webcamId_pmRange = np.load(DEV_PATH_TO_HAZE_FEATURES_NPY_BY_WEBCAM_ID_AND_PMRANGE_PREFIX +  str(webcamId_) + '_pmRange_' + str(current_pmRange_) + '.npy')[1:].reshape(-1, num_of_columns_for_haze_feature_matrix)
        TEST_dataset_weather_by_webcamId_pmRange = np.load(TEST_PATH_TO_WEATHER_NPY_BY_WEBCAM_ID_AND_PMRANGE_PREFIX +  str(webcamId_) + '_pmRange_' + str(current_pmRange_) + '.npy').reshape(-1, num_of_columns_for_weather_feature_matrix)
        TEST_dataset_hazefeatures_by_webcamId_pmRange = np.load(TEST_PATH_TO_HAZE_FEATURES_NPY_BY_WEBCAM_ID_AND_PMRANGE_PREFIX +  str(webcamId_) + '_pmRange_' + str(current_pmRange_) + '.npy')[1:].reshape(-1, num_of_columns_for_haze_feature_matrix)
        
        if ISNONE_entire_dataset_by_webcam_id_with_haze_features:
            ISNONE_entire_dataset_by_webcam_id_with_haze_features = False
            entire_dataset_by_webcam_id_with_haze_features = np.concatenate((TRAIN_dataset_hazefeatures_by_webcamId_pmRange, DEV_dataset_hazefeatures_by_webcamId_pmRange, TEST_dataset_hazefeatures_by_webcamId_pmRange), axis=0)
        else:
            entire_dataset_by_webcam_id_with_haze_features = np.concatenate((entire_dataset_by_webcam_id_with_haze_features, TRAIN_dataset_hazefeatures_by_webcamId_pmRange, DEV_dataset_hazefeatures_by_webcamId_pmRange, TEST_dataset_hazefeatures_by_webcamId_pmRange), axis=0)
        
        if ISNONE_entire_dataset_by_webcam_id_with_feature_features:
            ISNONE_entire_dataset_by_webcam_id_with_feature_features = False
            entire_dataset_by_webcam_id_with_weather_features = np.concatenate((TRAIN_dataset_weather_by_webcamId_pmRange, DEV_dataset_weather_by_webcamId_pmRange, TEST_dataset_weather_by_webcamId_pmRange), axis=0)
        else:
            entire_dataset_by_webcam_id_with_weather_features = np.concatenate((entire_dataset_by_webcam_id_with_weather_features, TRAIN_dataset_weather_by_webcamId_pmRange, DEV_dataset_weather_by_webcamId_pmRange, TEST_dataset_weather_by_webcamId_pmRange), axis=0)
    entire_dataset_by_webcam_id_with_haze_features_df = pd.DataFrame(entire_dataset_by_webcam_id_with_haze_features, columns=column_names_with_haze_features)
    entire_dataset_by_webcam_id_with_weather_features_df = pd.DataFrame(entire_dataset_by_webcam_id_with_weather_features, columns=column_names_with_weather_features)
    entire_dataset_by_webcam_id_with_all_features_df = pd.merge(entire_dataset_by_webcam_id_with_haze_features_df, entire_dataset_by_webcam_id_with_weather_features_df, how='left', left_on=['filePath'], right_on=['filePath'])
    entire_dataset_by_webcam_id_with_all_features_df = entire_dataset_by_webcam_id_with_all_features_df.drop(columns=['webcamId_y', 'webcamLat_y', 'webcamLong_y', 'year_y', 'date_y', 'hour_y', 'range_y', 'pm_y'])
    entire_dataset_by_webcam_id_with_all_features_npy = entire_dataset_by_webcam_id_with_all_features_df.as_matrix()
    np.save(PATH_TO_SAVE_DATASET_BY_WEBCAMID_OF_HAZE_AND_WEATHER_FEATURES_PREFIX + str(webcamId_), entire_dataset_by_webcam_id_with_all_features_npy)

def create_npy_of_with_only_metadata(webcamId_):

def create_train_dev_test_temporal_split_from_webcamId(webcamId_):
    dataset_by_webcamId_ = np.load(PATH_TO_SAVE_DATASET_BY_WEBCAMID_OF_HAZE_AND_WEATHER_FEATURES_PREFIX + str(webcamId_) + ".npy")
    df_dataset_by_webcamId = pd.DataFrame(dataset_by_webcamId_, columns=all_column_names)
    df_dataset_by_webcamId['year'] = df_dataset_by_webcamId['year'].astype('int32')
    df_dataset_by_webcamId['webcamId'] = df_dataset_by_webcamId['webcamId'].astype('int32')
    df_dataset_by_webcamId['hour'] = df_dataset_by_webcamId['hour'].str.strip('.jpg')
    df_dataset_by_webcamId['hour'] = df_dataset_by_webcamId['hour'].astype('int32')
    num_of_total_examples = len(df_dataset_by_webcamId.index)
    df_dataset_by_webcamId_training_set  = df_dataset_by_webcamId[df_dataset_by_webcamId['year'] <= 2014]
    df_dataset_by_webcamId_non_training_set = df_dataset_by_webcamId[df_dataset_by_webcamId['year'] > 2014]
    num_of_training_examples = len(df_dataset_by_webcamId_training_set.index)
    num_of_non_training_examples = len(df_dataset_by_webcamId_non_training_set.index)
    print(num_of_total_examples, num_of_training_examples, num_of_non_training_examples)
    years = [2009, 2010, 2011, 2012, 2013, 2014]
    pmRanges = ['00-09', '10-19', '20-29', '30-39', '40-49', '50-59']
    print("PRINTING COUNTS FOR TRAINING SEGMENTATIONS")
    for current_year_ in years:
        for current_pmRange_ in pmRanges:
            current_segmentation_of_training_dataset_by_webcamId_year_and_pmRange = df_dataset_by_webcamId_training_set[df_dataset_by_webcamId_training_set['year'] == current_year_]
            current_segmentation_of_training_dataset_by_webcamId_year_and_pmRange = current_segmentation_of_training_dataset_by_webcamId_year_and_pmRange[current_segmentation_of_training_dataset_by_webcamId_year_and_pmRange['range'] == current_pmRange_]
            print(current_year_, current_pmRange_, len(current_segmentation_of_training_dataset_by_webcamId_year_and_pmRange.index))
            current_segmentation_of_training_dataset_by_webcamId_year_and_pmRange_npy = current_segmentation_of_training_dataset_by_webcamId_year_and_pmRange.as_matrix()
            np.save(PATH_TO_TRAIN_DATASET_OF_HAZE_AND_WEATHER_FEATURES_BY_WEBCAMID_YEAR_AND_PMRANGE_PREFIX + str(webcamId_) + "_year_" + str(current_year_) + "_pmRange_" + str(current_pmRange_), current_segmentation_of_training_dataset_by_webcamId_year_and_pmRange_npy)
    print("END OF COUNTS FOR TRAINING SEGMENTATION")

    shuffled_indices = np.arange(num_of_non_training_examples)
    np.random.shuffle(shuffled_indices)
    dataset_by_webcamId_non_training_set_npy = df_dataset_by_webcamId_non_training_set.as_matrix()
    shuffled_original_dataset = dataset_by_webcamId_non_training_set_npy[shuffled_indices]
    
    dev_dataset_examples = shuffled_original_dataset[:int((num_of_non_training_examples*1.0) / 2.0)]
    test_dataset_examples = shuffled_original_dataset[int((num_of_non_training_examples*1.0) / 2.0):]


    print("SHOULD HAVE THE FOLLOWING LENGTHS")
    print(num_of_non_training_examples)
    print(int((num_of_non_training_examples * 1.0) / 2.0))
    print("END OF - SHOULD HAVE THE FOLLOWING LENGTHS")
    print("Actual Lengths")
    print(len(shuffled_original_dataset))
    print(len(dev_dataset_examples))
    print(len(test_dataset_examples))
    np.save(PATH_TO_DEV_DATASET_OF_HAZE_AND_WEATHER_FEATURES_BY_WEBCAMID_PREFIX + str(webcamId_), dev_dataset_examples)
    np.save(PATH_TO_TEST_DATASET_OF_HAZE_AND_WEATHER_FEATURES_BY_WEBCAMID_PREFIX + str(webcamId_), test_dataset_examples)

def create_feature(current_example):
    img_npy = imread("/mnt" + current_example[0])
    img_npy = skimage.transform.resize(img_npy, (500, 500, 3))
    img_npy = np.array(img_npy).astype(np.float64)
    print(img_npy.shape)
    ex_feature = {'image': _bytes_feature(np.array(img_npy).tostring()),
                    'orig_height': _int64_feature(img_npy.shape[0]),
                    'orig_width': _int64_feature(img_npy.shape[1]),
                    'filePath': _bytes_feature(current_example[0].encode("utf-8")), 
                    'webcamId': _int64_feature(int(current_example[1])),
                    'webcamLat': _float_feature(float(current_example[2])),
                    'webcamLong': _float_feature(float(current_example[3])),
                    'year': _int64_feature(int(current_example[4])),
                    'date': _int64_feature(int(current_example[5])),
                    'hour': _int64_feature(int(current_example[6]))}
    for _current_column_name_, _current_column_val_ in zip(all_column_names[8:], current_example[8:]):
        ex_feature[_current_column_name_] = _float_feature(float(_current_column_val_))
    return ex_feature

def create_TRAIN_tf_records_by_webcamId_year_and_pmRange(webcamId_, year_, pmRange_):
    TRAIN_dataset_by_webcamId_year_and_pmRange = np.load(PATH_TO_TRAIN_DATASET_OF_HAZE_AND_WEATHER_FEATURES_BY_WEBCAMID_YEAR_AND_PMRANGE_PREFIX + str(webcamId_) + "_year_" + str(year_) + "_pmRange_" + str(pmRange_) + ".npy")
    TRAIN_tfrecord_with_all_features = '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features_code_submission/TRAIN_tfrecord_with_all_features_' + str(webcamId_) + "_year_" + str(year_) + "_pmRange_" + str(pmRange_) + '.tfrecord'

    TRAIN_writer = tf.python_io.TFRecordWriter(TRAIN_tfrecord_with_all_features) 
    counter = 1
    for TRAIN_current_example_in_dataset in TRAIN_dataset_by_webcamId_year_and_pmRange:
        try:
            ex_feature = create_feature(TRAIN_current_example_in_dataset)
            example = tf.train.Example(features=tf.train.Features(feature=ex_feature))
            print('TRAIN TFRECORD: ' + TRAIN_current_example_in_dataset[0], webcamId_, year_, pmRange_, counter)
            counter += 1
            TRAIN_writer.write(example.SerializeToString())
        except:
            continue

def create_DEV_tf_records_by_webcamId(webcamId_):
    DEV_dataset_by_webcamId = np.load(PATH_TO_DEV_DATASET_OF_HAZE_AND_WEATHER_FEATURES_BY_WEBCAMID_PREFIX + str(webcamId_) + ".npy")
    DEV_tfrecord_with_all_features = '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features_code_submission/DEV_tfrecord_with_all_features_' + str(webcamId_) + '.tfrecord'

    DEV_writer = tf.python_io.TFRecordWriter(DEV_tfrecord_with_all_features)
    counter = 1
    for DEV_current_example_in_dataset in DEV_dataset_by_webcamId:
        try:
            ex_feature = create_feature(DEV_current_example_in_dataset)
            example = tf.train.Example(features=tf.train.Features(feature=ex_feature))
            print('DEV TFRECORD: ' + DEV_current_example_in_dataset[0], webcamId_, counter)
            counter += 1
            DEV_writer.write(example.SerializeToString())
        except:
            continue 
    

def create_TEST_tf_records_by_webcamId(webcamId_):
    TEST_dataset_by_webcamId = np.load(PATH_TO_TEST_DATASET_OF_HAZE_AND_WEATHER_FEATURES_BY_WEBCAMID_PREFIX + str(webcamId_) + ".npy")
    TEST_tfrecord_with_all_features = '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features_code_submission/TEST_tfrecord_with_all_features_' + str(webcamId_) + '.tfrecord'

    TEST_writer = tf.python_io.TFRecordWriter(TEST_tfrecord_with_all_features)
    counter = 1
    for TEST_current_example_in_dataset in TEST_dataset_by_webcamId:
        try:
            ex_feature = create_feature(TEST_current_example_in_dataset)
            example = tf.train.Example(features=tf.train.Features(feature=ex_feature))
            print('TEST TFRECORD: ' + TEST_current_example_in_dataset[0], webcamId_, counter)
            counter += 1
            TEST_writer.write(example.SerializeToString())
        except:
            continue 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--webcamId', type=str)
    parser.add_argument('--year', type=str)
    parser.add_argument('--pmRange', type=str)
    parser.add_argument('--type', type=str)
    parser.add_argument('--mounted_root_directory', type=str)
    parser.add_argument('--directory_to_save_datasets_with_all_features', type=str)
    args = parser.parse_args()
    if args.type == "TRAIN":
        create_TRAIN_tf_records_by_webcamId_year_and_pmRange(args.webcamId, args.year, args.pmRange)
    elif args.type =="DEV":
        create_DEV_tf_records_by_webcamId(args.webcamId)
    else:
        create_TEST_tf_records_by_webcamId(args.webcamId)


