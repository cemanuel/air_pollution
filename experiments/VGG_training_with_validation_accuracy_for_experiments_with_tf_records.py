import AQP_experiment_class as AQP
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.python.client import timeline
import time
import numpy as np


experiment_directory_suffix = "VGG_experiment_final_presentation_webcamId_17603_with_regression_timesplit_with_3100_steps"
experiment_directory_name = None

dict_from_webcamId_to_filePath_to_num_of_examples_in_tfrecord = {
                                                                    '1066': {
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_1066_year_2009_pmRange_00-09.tfrecord': 4260,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_1066_year_2009_pmRange_10-19.tfrecord': 773,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_1066_year_2009_pmRange_20-29.tfrecord': 221,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_1066_year_2009_pmRange_30-39.tfrecord': 49,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_1066_year_2009_pmRange_40-49.tfrecord': 6,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_1066_year_2009_pmRange_50-59.tfrecord': 4,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_1066_year_2010_pmRange_00-09.tfrecord': 2830,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_1066_year_2010_pmRange_10-19.tfrecord': 314,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_1066_year_2010_pmRange_20-29.tfrecord': 93,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_1066_year_2010_pmRange_30-39.tfrecord': 14,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_1066_year_2011_pmRange_00-09.tfrecord': 1911,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_1066_year_2011_pmRange_10-19.tfrecord': 245,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_1066_year_2011_pmRange_20-29.tfrecord': 44,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_1066_year_2011_pmRange_30-39.tfrecord': 4,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_1066_year_2012_pmRange_00-09.tfrecord': 2626,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_1066_year_2012_pmRange_10-19.tfrecord': 283,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_1066_year_2012_pmRange_20-29.tfrecord': 85,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_1066_year_2012_pmRange_30-39.tfrecord': 24,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_1066_year_2012_pmRange_40-49.tfrecord': 11,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_1066_year_2013_pmRange_00-09.tfrecord': 3170,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_1066_year_2013_pmRange_10-19.tfrecord': 424,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_1066_year_2013_pmRange_20-29.tfrecord': 57,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_1066_year_2013_pmRange_30-39.tfrecord': 4,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_1066_year_2013_pmRange_40-49.tfrecord': 4,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_1066_year_2013_pmRange_50-59.tfrecord': 1,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_1066_year_2014_pmRange_00-09.tfrecord': 3399,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_1066_year_2014_pmRange_10-19.tfrecord': 401,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_1066_year_2014_pmRange_20-29.tfrecord': 70,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_1066_year_2014_pmRange_30-39.tfrecord': 18,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_1066_year_2014_pmRange_40-49.tfrecord': 2,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_1066_year_2014_pmRange_50-59.tfrecord': 4,
                                                                    },
                                                                    '17603': {
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_17603_year_2012_pmRange_00-09.tfrecord': 415,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_17603_year_2012_pmRange_10-19.tfrecord': 461,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_17603_year_2012_pmRange_20-29.tfrecord': 107,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_17603_year_2012_pmRange_30-39.tfrecord': 32,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_17603_year_2012_pmRange_40-49.tfrecord': 7,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_17603_year_2012_pmRange_50-59.tfrecord': 10,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_17603_year_2013_pmRange_00-09.tfrecord': 1030,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_17603_year_2013_pmRange_10-19.tfrecord': 992,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_17603_year_2013_pmRange_20-29.tfrecord': 254,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_17603_year_2013_pmRange_30-39.tfrecord': 38,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_17603_year_2013_pmRange_40-49.tfrecord': 2,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_17603_year_2014_pmRange_00-09.tfrecord': 7660,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_17603_year_2014_pmRange_10-19.tfrecord': 9478,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_17603_year_2014_pmRange_20-29.tfrecord': 2008,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_17603_year_2014_pmRange_30-39.tfrecord': 350,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_17603_year_2014_pmRange_40-49.tfrecord': 8,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_17603_year_2014_pmRange_50-59.tfrecord': 12,
                                                                    },
                                                                    '21587': {
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_21587_year_2012_pmRange_00-09.tfrecord': 1023,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_21587_year_2012_pmRange_10-19.tfrecord': 738,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_21587_year_2012_pmRange_20-29.tfrecord': 333,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_21587_year_2012_pmRange_30-39.tfrecord': 86,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_21587_year_2012_pmRange_40-49.tfrecord': 32,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_21587_year_2012_pmRange_50-59.tfrecord': 5,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_21587_year_2013_pmRange_00-09.tfrecord': 4795,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_21587_year_2013_pmRange_10-19.tfrecord': 4672,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_21587_year_2013_pmRange_20-29.tfrecord': 1739,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_21587_year_2013_pmRange_30-39.tfrecord': 425,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_21587_year_2013_pmRange_40-49.tfrecord': 109,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_21587_year_2014_pmRange_00-09.tfrecord': 2011,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_21587_year_2014_pmRange_10-19.tfrecord': 2103,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_21587_year_2014_pmRange_20-29.tfrecord': 740,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_21587_year_2014_pmRange_30-39.tfrecord': 228,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_21587_year_2014_pmRange_40-49.tfrecord': 65,
                                                                                '/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features/TRAIN_tfrecord_with_all_features_21587_year_2014_pmRange_50-59.tfrecord': 22,
                                                                    }
                                                                }

dict_from_webcamId_to_total_num_of_training_examples = {
                                                            '1066': 21351,
                                                            '17603': 22864,
                                                            '21587': 1914,
                                                        }


def initialize_params_for_experiment():
    params_initialization_for_training = {}
    params_initialization_for_training['training_path'] = "/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features"
    params_initialization_for_training['logs_dir'] = "/mnt/mnt/mounted_bucket/results_for_final_presentation/logs"
    params_initialization_for_training['summary_dir'] = "/mnt/mnt/mounted_bucket/results_for_final_presentation/summary"
    params_initialization_for_training['webcamId'] = '17603'
    params_initialization_for_training['dict_of_filePath_to_num_of_examples_in_tfrecord'] = dict_from_webcamId_to_filePath_to_num_of_examples_in_tfrecord[params_initialization_for_training['webcamId']]
    params_initialization_for_training['total_num_of_training_examples'] = dict_from_webcamId_to_total_num_of_training_examples[params_initialization_for_training['webcamId']]
    params_initialization_for_training['num_of_classes'] = 1
    params_initialization_for_training['min_bins'] = None
    params_initialization_for_training['max_bins'] = None
    params_initialization_for_training['batch_size'] = 32
    params_initialization_for_training['batch_size_per_tfrecord'] = None
    params_initialization_for_training['stage_of_development'] = "training"
    params_initialization_for_training['max_num_epochs'] = 4
    params_initialization_for_training['type_of_model'] = 'VGG'
    params_initialization_for_training['model_path'] = "/mnt/mnt/mounted_bucket/pretrained_weights/vgg/vgg_16.ckpt"
    params_initialization_for_training['type_of_optimizer'] = 'adam'
    params_initialization_for_training['beta1'] = 0.9
    params_initialization_for_training['beta2'] = 0.999    
    params_initialization_for_training['initial_num_steps'] = 50
    params_initialization_for_training['initial_num_partial_steps'] = 50
    params_initialization_for_training['initial_learning_rate'] = 0.0001
    params_initialization_for_training['initial_partial_learning_rate'] = 0.0001
    params_initialization_for_training['learning_rate_decay_factor'] = 0.99
    params_initialization_for_training['num_steps'] = 1000
    params_initialization_for_training['num_partial_steps'] = 0

    return AQP.initialize_params_tf_records(experiment_directory_name,
                                                experiment_directory_suffix,
                                                "training",
                                                params_initialization_for_training=params_initialization_for_training,
                                                training_with_eval=True)

params, list_of_tf_records_for_training, list_of_tf_records_for_dev = initialize_params_for_experiment()
for tf_record_example_for_training_ in list_of_tf_records_for_training:
    print(tf_record_example_for_training_)
for tf_record_example_for_dev_ in list_of_tf_records_for_dev:
    print(tf_record_example_for_dev_)
if params != None and list_of_tf_records_for_training != None and list_of_tf_records_for_dev != None:
    AQP.run_training_with_tfrecords_and_validation(params,
                                                    "/gpu:0",
                                                    list_of_tf_records_for_training,
                                                    list_of_tf_records_for_dev)
