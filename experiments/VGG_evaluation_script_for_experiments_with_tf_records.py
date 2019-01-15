import AQP_experiment_class as AQP
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.python.client import timeline
import time
import numpy as np
import cv2

experiment_directory_suffix = "VGG_experiment_final_presentation_webcamId_17603_with_regression_timesplit_with_3100_steps"
experiment_directory_name = None


def initialize_params_for_evaluation_from_checkpoint():
    params_initialization_for_evaluation = {}
    params_initialization_for_evaluation['logs_dir'] = "/mnt/mnt/mounted_bucket/results_for_final_presentation/logs"
    params_initialization_for_evaluation['summary_dir'] = "/mnt/mnt/mounted_bucket/results_for_final_presentation/summary"
    params_initialization_for_evaluation['evaluate_model'] = True
    params_initialization_for_evaluation['resume_training'] = False
    params_initialization_for_evaluation['stage_of_development'] = "evaluation"
    params_initialization_for_evaluation['type_of_evaluation'] = 'DEV'
    params_initialization_for_evaluation['training_path'] = "/mnt/mnt/mounted_bucket/tfRecord_datasets_with_all_features"
    params_initialization_for_evaluation['data_dir'] = 'data'
    params_initialization_for_evaluation['max_steps'] = None
    params_initialization_for_evaluation['num_epochs'] = 1
    params_initialization_for_evaluation['batch_size'] = 32
    params_initialization_for_evaluation['num_steps'] = 300
    params_initialization_for_evaluation['pmRange'] = None
    params_initialization_for_evaluation['year'] = None
    params_initialization_for_evaluation['type_of_model'] = 'VGG'

    return AQP.initialize_params_tf_records(experiment_directory_name,
                                    experiment_directory_suffix,
                                    "evaluation",
                                    params_initialization_for_evaluation=params_initialization_for_evaluation,
                                    use_ranges=True)


params, list_of_tf_records_for_evaluation, _ = initialize_params_for_evaluation_from_checkpoint()
for tf_record_for_evaluation_ in list_of_tf_records_for_evaluation:
    print(tf_record_for_evaluation_)
if params != None and list_of_tf_records_for_evaluation:
    AQP.run_evaluation_tfrecords(params,
                                    "/gpu:0",
                                    list_of_tf_records_for_evaluation)