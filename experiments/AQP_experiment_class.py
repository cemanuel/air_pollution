import data_utils
from tensorflow.python.platform import gfile
import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator
import sys
import cv2
sys.path.append("../")
from models.aqp_model_template import AQPModel
import math
import statistics
import numpy as np
import pickle
import os
import re
import datetime
import time
import imghdr

WEBCAMIDS = ['20265', '13342', '20258', '18879', '17603', '21587', '21673', '1066', '2045', '93', '5207', '6629', '1333', '30', '204', '22014']
PMRANGES = ["00-09", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69"]


def initialize_params_helper_tf_records(params,
                                        stage_of_development,
                                        experiment_dir_name,
                                        experiment_dir_suffix,
                                        params_initialization_for_training=None,
                                        params_initialization_for_resume_training=None,
                                        params_initialization_for_evaluation=None):
    filenames_of_images_zero_labeled = None
    labels_of_images_zero_labeled = None
    filenames_of_images_one_labeled = None
    labels_of_images_one_labeled = None
    filenames_of_images = []
    labels_of_images = []

    list_of_filenames_of_tfrecords = []

    if stage_of_development == 'training':
        for key_parameter in params_initialization_for_training:
            params[key_parameter] = params_initialization_for_training[key_parameter]
        directories_to_create_experiment_folders_in = [params['logs_dir'], params['summary_dir']]
        directories_to_search_in = [params['logs_dir'], params['summary_dir']]
        params['experiment_dir'] = data_utils.create_experiment_folders(experiment_dir_suffix, directories_to_create_experiment_folders_in)
        print(params['experiment_dir'])
        if params['experiment_dir'] == None:
            return None, None, None
        if params['webcamId'] != None:
            list_of_filtered_filenames_from_training_path = [f for f in os.listdir(params['training_path']) if re.match(r'TRAIN_*', f) and re.match('.*' + params['webcamId'] + '.*', f) and os.path.getsize(params['training_path']  + "/" + f) > 0]
        else:
            list_of_filtered_filenames_from_training_path = [f for f in os.listdir(params['training_path']) if re.match(r'TRAIN_*', f) and os.path.getsize(params['training_path'] + "/" + f) > 0]
        list_of_filenames_of_tfrecords.extend(list_of_filtered_filenames_from_training_path)
        if params['webcamId'] != None:
            list_of_tfrecords_from_training_path = [params['training_path'] + "/" + f for f in list_of_filtered_filenames_from_training_path if re.match(r'TRAIN_*', f) and re.match('.*' + params['webcamId'] + '.*', f) and os.path.getsize(params['training_path'] + "/" + f) > 0]
        else:
            list_of_tfrecords_from_training_path = [params['training_path'] + "/" + f for f in list_of_filtered_filenames_from_training_path if re.match(r'TRAIN_*', f) and os.path.getsize(params['training_path'] + "/" + f) > 0]

        list_of_tf_records = []
        list_of_tf_records.extend(list_of_tfrecords_from_training_path)
        file_to_save_pickle = params['logs_dir'] + "/" + params['experiment_dir'] + "/" + "parameters.pickle"
        pickle.dump(params, open(file_to_save_pickle, "wb"))
    else:
        if stage_of_development == "resume_training":
            for key_parameter in params_initialization_for_resume_training:
                params[key_parameter] = params_initialization_for_resume_training[key_parameter]
        else:
            for key_parameter in params_initialization_for_evaluation:
                params[key_parameter] = params_initialization_for_evaluation[key_parameter]
        directories_to_create_experiment_folders_in = [params['logs_dir'], params['summary_dir']]
        directories_to_search_in = [params['logs_dir'], params['summary_dir']]
        experiment_directory_name = data_utils.find_experiment_name(experiment_dir_name, experiment_dir_suffix, directories_to_search_in)
        if experiment_directory_name == None:
            return None, None, None
        if experiment_directory_name == None:
            print("There was an error in identifying the experiment directory to evaluate model.")
            print("Please select an experiment that has been trained with a model.")
            return
        if params['logs_dir'] == None:
            print("Please provide the path to the logs directory that stores pickle files of saved parameters of different experiment runs.")
            return
        params['experiment_dir'] = experiment_directory_name
        file_to_load_pickle = params['logs_dir'] + "/" + params['experiment_dir']  + "/" + "parameters.pickle"
        if not os.path.isfile(file_to_load_pickle):
            print(params['logs_dir'] + "/" + params['experiment_dir']  + "/" + "parameters.pickle")
            print("Parameters pickel file does not exist.")
            print("Please select an experiment that has been trained with a model.")
            return
        params = pickle.load(open(file_to_load_pickle, "rb"))
        if stage_of_development == 'resume_training':
            params['resume_training'] = True
            for key_parameter in params_initialization_for_resume_training:
                params[key_parameter] = params_initialization_for_resume_training[key_parameter]
            directories_to_create_experiment_folders_in = [params['logs_dir'], params['summary_dir']]
            directories_to_search_in = [params['logs_dir'], params['summary_dir']]
            if params['webcamId'] != None:
                list_of_filtered_filenames_from_training_path = [f for f in os.listdir(params['training_path']) if re.match(r'TRAIN_*', f) and re.match('.*' + params['webcamId'] + '*', f) and os.path.getsize(params['training_path']  + "/" + f) > 0]
            else:
                list_of_filtered_filenames_from_training_path = [f for f in os.listdir(params['training_path']) if re.match(r'TRAIN_*', f) and os.path.getsize(params['training_path'] + "/" + f) > 0]
            list_of_filenames_of_tfrecords.extend(list_of_filtered_filenames_from_training_path)
            if params['webcamId'] != None:
                list_of_tfrecords_from_training_path = [params['training_path'] + "/" + f for f in list_of_filtered_filenames_from_training_path if re.match(r'TRAIN_*', f) and re.match('.*' + params['webcamId'] + '*', f) and os.path.getsize(params['training_path'] + "/" + f) > 0]
            else:
                list_of_tfrecords_from_training_path = [params['training_path'] + "/" + f for f in list_of_filtered_filenames_from_training_path if re.match(r'TRAIN_*', f) and os.path.getsize(params['training_path'] + "/" + f) > 0]
            list_of_tf_records = []
            list_of_tf_records.extend(list_of_tfrecords_from_training_path)
        else:
            params['forward_only'] = True
            params['evaluate_model'] = True
            for key_parameter in params_initialization_for_evaluation:
                params[key_parameter] = params_initialization_for_evaluation[key_parameter]
            directories_to_create_experiment_folders_in = [params['logs_dir'], params['summary_dir']]
            directories_to_search_in = [params['logs_dir'], params['summary_dir']]
            if params['webcamId'] != None and params['pmRange'] != None and params['year']:
                list_of_filtered_filenames_from_training_path = [f for f in os.listdir(params['training_path']) if re.match(params['type_of_evaluation'] + '_*', f) and re.match('.*' + params['webcamId'] + '.*', f) and re.match('.*' + params['pmRange'] + '.*', f) and re.match('.*' + params['year'] + '.*', f) and os.path.getsize(params['training_path']  + "/" + f) > 0]
            elif params['webcamId'] != None:
                list_of_filtered_filenames_from_training_path = [f for f in os.listdir(params['training_path']) if re.match(params['type_of_evaluation'] + '_*', f) and re.match('.*' + params['webcamId'] + '.*', f) and os.path.getsize(params['training_path']  + "/" + f) > 0]
            elif params['pmRange'] != None:
                list_of_filtered_filenames_from_training_path = [f for f in os.listdir(params['training_path']) if re.match(params['type_of_evaluation'] + '_*', f) and re.match('.*' + params['pmRange'] + '.*', f) and os.path.getsize(params['training_path']  + "/" + f) > 0]
            elif params['year'] != None:
                list_of_filtered_filenames_from_training_path = [f for f in os.listdir(params['training_path']) if re.match(params['type_of_evaluation'] + '_*', f) and re.match('.*' + params['year'] + '.*', f) and os.path.getsize(params['training_path']  + "/" + f) > 0]
            else:
                list_of_filtered_filenames_from_training_path = [f for f in os.listdir(params['training_path']) if re.match(params['type_of_evaluation'] + '_*', f) and os.path.getsize(params['training_path']  + "/" + f) > 0]
            list_of_filenames_of_tfrecords.extend(list_of_filtered_filenames_from_training_path)
            if params['webcamId'] != None and params['pmRange'] != None and params['year'] != None:
                list_of_tfrecords_from_training_path = [params['training_path'] + "/" + f for f in list_of_filtered_filenames_from_training_path if re.match(params['type_of_evaluation'] + '_*', f) and re.match('.*' + params['webcamId'] + '.*', f) and re.match('.*' + params['pmRange'] + '.*', f) and os.path.getsize(params['training_path'] + "/" + f) > 0]
            elif params['webcamId'] != None:
                list_of_tfrecords_from_training_path = [params['training_path'] + "/" + f for f in list_of_filtered_filenames_from_training_path if re.match(params['type_of_evaluation'] + '_*', f) and re.match('.*' + params['webcamId'] + '.*', f) and os.path.getsize(params['training_path'] + "/" + f) > 0]
            elif params['pmRange'] != None:
                list_of_tfrecords_from_training_path = [params['training_path'] + "/" + f for f in list_of_filtered_filenames_from_training_path if re.match(params['type_of_evaluation'] + '_*', f) and re.match('.*' + params['pmRange'] + '.*', f) and os.path.getsize(params['training_path'] + "/" + f) > 0]
            elif params['year'] != None:
                list_of_tfrecords_from_training_path = [params['training_path'] + "/" + f for f in list_of_filtered_filenames_from_training_path if re.match(params['type_of_evaluation'] + '_*', f) and re.match('.*' + params['year'] + '.*', f) and os.path.getsize(params['training_path'] + "/" + f) > 0]
            else:
                list_of_tfrecords_from_training_path = [params['training_path'] + "/" + f for f in list_of_filtered_filenames_from_training_path if re.match(params['type_of_evaluation'] + '_*', f) and os.path.getsize(params['training_path'] + "/" + f) > 0]
            list_of_tf_records = []
            list_of_tf_records.extend(list_of_tfrecords_from_training_path)
    return params, list_of_tf_records

def initialize_params_tf_records(experiment_dir_name,
                                    experiment_dir_suffix,
                                    stage_of_development,
                                    params_initialization_for_training=None,
                                    params_initialization_for_resume_training=None,
                                    params_initialization_for_evaluation=None,
                                    training_with_eval=False,
                                    use_ranges=False):
    params = {}
    params['experiment_dir'] = None
    params['resume_training'] = False
    params['evaluate_model'] = False
    params['resume_training'] = False
    params['num_epochs'] = None
    params['max_steps'] = None
    params['num_steps_before_checkpoint'] = 1
    params['data_dir'] = None
    params['logs_dir'] = None
    params['summary_dir'] = None
    params['results_dir'] = None
    params['training_path'] = None
    params['dev_path'] = None
    params['testing_path'] = None
    params['forward_only'] = False
    params['use_preprocessing'] = False
    params['model_path'] = None
    params['training_with_eval'] = training_with_eval

    params, list_of_tf_records = initialize_params_helper_tf_records(params,
                                                                        stage_of_development,
                                                                        experiment_dir_name,
                                                                        experiment_dir_suffix,
                                                                        params_initialization_for_training=params_initialization_for_training,
                                                                        params_initialization_for_resume_training=params_initialization_for_resume_training,
                                                                        params_initialization_for_evaluation=params_initialization_for_evaluation)
    list_of_tf_records_eval = None
    if stage_of_development == "training" or stage_of_development == "resume_training":
        if training_with_eval:
            list_of_filenames_of_tfrecords = []
            if params['webcamId'] != None:
                list_of_filtered_filenames_from_training_path = [f for f in os.listdir(params['training_path']) if re.match(r'DEV_*', f) and re.match('.*' + params['webcamId'] + '.*', f) and os.path.getsize(params['training_path']  + "/" + f) > 0]
            else:
                list_of_filtered_filenames_from_training_path = [f for f in os.listdir(params['training_path']) if re.match(r'DEV_*', f) and os.path.getsize(params['training_path']  + "/" + f) > 0]
            list_of_filenames_of_tfrecords.extend(list_of_filtered_filenames_from_training_path)
            if params['webcamId'] != None:
                list_of_tfrecords_from_training_path = [params['training_path'] + "/" + f for f in list_of_filtered_filenames_from_training_path if re.match(r'DEV_*', f) and re.match('.*' + params['webcamId'] + '.*', f) and os.path.getsize(params['training_path'] + "/" + f) > 0]
            else:
                list_of_tfrecords_from_training_path = [params['training_path'] + "/" + f for f in list_of_filtered_filenames_from_training_path if re.match(r'DEV_*', f) and os.path.getsize(params['training_path'] + "/" + f) > 0]
            list_of_tf_records_eval = []
            list_of_tf_records_eval.extend(list_of_tfrecords_from_training_path)
            
    return params, list_of_tf_records, list_of_tf_records_eval

def initialize_params_helper(params, 
                                stage_of_development,
                                experiment_dir_name,
                                experiment_dir_suffix,
                                params_initialization_for_training=None,
                                params_initialization_for_resume_training=None,
                                params_initialization_for_evaluation=None):

    directories_to_create_experiment_folders_in = ['logs', 'summary']
    directories_to_search_in = ['logs', 'summary']

    filenames_of_images_zero_labeled = None
    labels_of_images_zero_labeled = None
    filenames_of_images_one_labeled = None
    labels_of_images_one_labeled = None
    filenames_of_images = []
    labels_of_images = []

    if stage_of_development == 'training':
        for key_parameter in params_initialization_for_training:
            params[key_parameter] = params_initialization_for_training[key_parameter]
        params['experiment_dir'] = data_utils.create_experiment_folders(experiment_dir_suffix, directories_to_create_experiment_folders_in)
        if params['experiment_dir'] == None:
            return None, None, None
        file_paths, webcamLats, webcamLongs, webcamIds, years, dates, hours, ranges_, pmValues = data_utils.prepare_data(params['training_path'])
        file_to_save_pickle = params['logs_dir'] + "/" + params['experiment_dir'] + "/" + "parameters.pickle"
        pickle.dump(params, open(file_to_save_pickle, "wb"))
    else:
        
        experiment_directory_name = data_utils.find_experiment_name(experiment_dir_name, experiment_dir_suffix, directories_to_search_in)
        if experiment_directory_name == None:
            return None, None, None
        if experiment_directory_name == None:
            print("There was an error in identifying the experiment directory to evaluate model.")
            print("Please select an experiment that has been trained with a model.")
            return
        if params['logs_dir'] == None:
            print("Please provide the path to the logs directory that stores pickle files of saved parameters of different experiment runs.")
            return
        params['experiment_dir'] = experiment_directory_name
        file_to_load_pickle = params['logs_dir'] + "/" + params['experiment_dir']  + "/" + "parameters.pickle"
        if not os.path.isfile(file_to_load_pickle):
            print(params['logs_dir'] + "/" + params['experiment_dir']  + "/" + "parameters.pickle")
            print("Parameters pickel file does not exist.")
            print("Please select an experiment that has been trained with a model.")
            return
        params = pickle.load(open(file_to_load_pickle, "rb"))
        if stage_of_development == 'resume_training':
            params['resume_training'] = True
            for key_parameter in params_initialization_for_resume_training:
                params[key_parameter] = params_initialization_for_resume_training[key_parameter]
            file_paths, webcamLats, webcamLongs, webcamIds, years, dates, hours, ranges_, pmValues = data_utils.prepare_data(params['training_path'])
        else:
            params['forward_only'] = True
            params['evaluate_model'] = True
            for key_parameter in params_initialization_for_evaluation:
                params[key_parameter] = params_initialization_for_evaluation[key_parameter]
            file_paths, webcamLats, webcamLongs, webcamIds, years, dates, hours, ranges_, pmValues = data_utils.prepare_data(params['dev_path'])
    return params, file_paths, pmValues

def initialize_params(experiment_dir_name,
                        experiment_dir_suffix,
                        stage_of_development,
                        params_initialization_for_training=None,
                        params_initialization_for_resume_training=None,
                        params_initialization_for_evaluation=None,
                        training_with_dev=False, 
                        use_ranges=False,):
    params = {}
    params['experiment_dir'] = None
    params['resume_training'] = False
    params['evaluate_model'] = False
    params['resume_training'] = False
    params['num_epochs'] = None
    params['max_steps'] = None
    params['num_steps_before_checkpoint'] = 1
    params['data_dir'] = None
    params['logs_dir'] = None
    params['summary_dir'] = None
    params['results_dir'] = None
    params['training_path'] = None
    params['dev_path'] = None
    params['testing_path'] = None
    params['forward_only'] = False
    params['use_preprocessing'] = False
    params['model_path'] = None
    params['dict_of_filePath_to_num_of_examples_in_tfrecord'] = None
    params['type_of_optimizer'] = 'adam'
    params['total_num_of_training_examples'] = None

    params, filenames_of_images, labels_of_images, = initialize_params_helper(params,
                                                                                stage_of_development,
                                                                                experiment_dir_name,
                                                                                experiment_dir_suffix,
                                                                                params_initialization_for_training=params_initialization_for_training,
                                                                                params_initialization_for_resume_training=params_initialization_for_resume_training,
                                                                                params_initialization_for_evaluation=params_initialization_for_evaluation)
    filenames_of_images_dev = None
    pmValues_dev = None
    if stage_of_development == "training" or stage_of_development == "resume_training":
        if training_with_dev:
            filenames_of_images_dev, _, _, _, _, _, _, _, pmValues_dev = data_utils.prepare_data(params['dev_path'])

    return params, filenames_of_images, labels_of_images,  filenames_of_images_dev, pmValues_dev

def create_model(session,
                    params,
                    list_of_tfrecords_for_training=None,
                    list_of_trecords_for_evaluation=None):
    model = AQPModel(session,
                        params['batch_size'],
                        params['stage_of_development'],
                        params['learning_rate_decay_factor'],
                        params['type_of_model'],
                        params['summary_dir'],
                        params['experiment_dir'],
                        params['type_of_optimizer'],
                        params['num_of_classes'],
                        params['total_num_of_training_examples'],
                        min_bins=params['min_bins'],
                        max_bins=params['max_bins'],
                        model_path=params['model_path'],
                        beta1=params['beta1'],
                        beta2=params['beta2'],
                        list_of_tfrecords_for_training=list_of_tfrecords_for_training,
                        list_of_tfrecords_for_evaluation=list_of_trecords_for_evaluation,
                        training_with_eval=params['training_with_eval'],
                        dict_of_filePath_to_num_of_examples_in_tfrecord=params['dict_of_filePath_to_num_of_examples_in_tfrecord'])
    ckpt = tf.train.get_checkpoint_state(params['logs_dir'] + "/" + params['experiment_dir'])
    if (ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path) and 
                (params['resume_training'] or params['evaluate_model'])):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        session.run(tf.tables_initializer())
        print("Finished initalizing tables")
        model.saver.restore(session, ckpt.model_checkpoint_path)
        print("Finished Restoring Model from checkpoint")
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())
        session.run(tf.tables_initializer())
    return model

def run_training_with_tfrecords_and_validation(params, gpu_device, list_of_tfrecords_for_training, list_of_tfrecords_for_evaluation):
    with tf.Graph().as_default(), tf.device(gpu_device):
        gpu_usage = 1.0
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_usage)
        with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)) as sess:
            model = create_model(sess, params, list_of_tfrecords_for_training=list_of_tfrecords_for_training, list_of_trecords_for_evaluation=list_of_tfrecords_for_evaluation)

            list_of_training_handles = []
            list_of_eval_handles = []

            for training_iterator_ in model.list_of_training_iterators:
                training_handle = sess.run(training_iterator_.string_handle())
                list_of_training_handles.append(training_handle)

            for eval_iterator_ in model.list_of_eval_iterators:
                eval_handle = sess.run(eval_iterator_.string_handle())
                list_of_eval_handles.append(eval_handle)

            eval_writer = tf.summary.FileWriter(params['summary_dir'] + '/' + params['experiment_dir'] + '/' + 'GLOBAL_EVAL', sess.graph)
            print("DEBUGGING - AFTER LIST OF GLOBAL EVAL STEPS/UPDATES")
            current_learning_rate = None
            current_partial_learning_rate = None
            for epoch_ in range(0, params['max_num_epochs']):
                if (params['type_of_model'] == 'VGG' or params['type_of_model'] == 'ResNet') and params['stage_of_development'] == "training" and epoch_ == 0:
                    print("DEBUGGING VGG INIT FN")
                    model.model.init_fn(sess)
                if params['type_of_model'] == 'DehazeNet' or params['type_of_model'] == 'VGG' or params['type_of_model'] == 'ResNet':
                    if params['type_of_model'] == 'VGG' and params['stage_of_development'] == "training" and epoch_ == 0:
                        sess.run(model.model.fc8_init)
                    if params['type_of_model'] == 'ResNet' and params['stage_of_development'] == "training" and epoch_ == 0:
                        sess.run(model.model.logits_init)
                    num_partial_steps_ = None
                    if epoch_ == 0:
                        num_partial_steps_ = params['initial_num_partial_steps']
                        current_learning_rate = params['initial_learning_rate']
                        current_partial_learning_rate = params['initial_partial_learning_rate']
                    else:
                        num_partial_steps_ = params['num_partial_steps']
                        current_learning_rate = current_learning_rate * params['learning_rate_decay_factor']
                        current_partial_learning_rate = current_partial_learning_rate * params['learning_rate_decay_factor']
                    for i_train_partial in range(num_partial_steps_):
                        try:
                            print("DEBUGGING TRAINING STEP")
                            start_time = time.time()
                            output_feed = [model.merged, model.MAE_, model.MSE_, model.MSLE_, model.R2_score_, model.row_indices, model.pm_values, model.batch_targets, model.predictions, model.partial_learning_rate, model.partial_train_op]
                            current_train_feed_dict = {}
                            for handle_idx_, current_handle_ in enumerate(model.list_of_handles):
                                current_train_feed_dict[current_handle_] = list_of_training_handles[handle_idx_]
                            current_train_feed_dict[model.partial_learning_rate] = current_partial_learning_rate
                            row_of_indices_for_batch_of_data = np.arange(model.batch_size)
                            np.random.shuffle(row_of_indices_for_batch_of_data)
                            current_train_feed_dict[model.row_indices] = row_of_indices_for_batch_of_data
                            current_train_feed_dict[model.is_training] = True
                            current_summary_train, current_MAE, current_MSE, current_MSLE, current_R2_SCORE, current_row_indices, current_pm_values, current_batch_targets, current_predictions, current_partial_learning_rate_for_session, _ = sess.run(output_feed, feed_dict=current_train_feed_dict)
                            end_time = time.time()
                            print("TRUE PM VALUES")
                            for pm_val_ in current_pm_values:
                                print(pm_val_)
                            print("END OF TRUE PM VALUES")
                            print("PREDICTED PM VALUES")
                            for pm_predicted_val_ in current_predictions:
                                print(pm_predicted_val_)
                            print("END OF PREDICTED PM VALUES")
                            model.train_writer.add_summary(current_summary_train, model.global_step.eval())
                            print("(TRAINING-PARTIAL) global step %d step-time %.2f MAE %.2f MSE %.2f MSLE %.2f R2 SCORE %.2f partial_learing_rate %f batch_size %f" % (model.global_step.eval(),
                                                                                                                                                                            end_time-start_time,
                                                                                                                                                                            current_MAE,
                                                                                                                                                                            current_MSE,
                                                                                                                                                                            current_MSLE,
                                                                                                                                                                            current_R2_SCORE,
                                                                                                                                                                            current_partial_learning_rate_for_session,
                                                                                                                                                                            model.batch_size))
                            checkpoint_path_directory = params['logs_dir'] + "/" + params['experiment_dir']
                            checkpoint_path = os.path.join(checkpoint_path_directory, "translate.ckpt")
                            '''
                            if model.global_step.eval() % params['num_steps_before_validation_checkpoint'] == 0 and model.global_step.eval() != 0:
                                for i_eval_ in range(params['num_steps_for_validation']):
                                    try:
                                        output_feed = [model.merged, model.MAE_, model.MSE_, model.MSLE_, model.R2_score_, model.row_indices, model.partial_learning_rate, model.global_eval_update_step_variable]
                                        current_eval_feed_dict = {}
                                        for handle_idx_, current_handle_ in enumerate(model.list_of_handles):
                                            current_eval_feed_dict[current_handle_] = list_of_eval_handles[handle_idx_]
                                        current_eval_feed_dict[model.partial_learning_rate] = current_partial_learning_rate
                                        row_of_indices_for_batch_of_data = np.arange(model.batch_size)
                                        np.random.shuffle(row_of_indices_for_batch_of_data)
                                        current_eval_feed_dict[model.row_indices] = row_of_indices_for_batch_of_data
                                        current_eval_feed_dict[model.is_training] = False
                                        current_summary_eval, current_MAE, current_MSE, current_MSLE, current_R2_SCORE, current_row_indices, current_partial_learning_fate_for_session, _ = sess.run(output_feed, feed_dict=current_eval_feed_dict)
                                        eval_writer.add_summary(current_summary_eval, model.global_eval_step.eval())
                                        print("(Evaluation - GLOBAL) GLOBAL STEP %d MAE %.2f MSE %.2f MSLE %.2f R2 SCORE %.2f partial_learning_rate %f batch_size %f" % (model.global_eval_step.eval(),
                                                                                                                                                                            current_MAE,
                                                                                                                                                                            current_MSE,
                                                                                                                                                                            current_MSLE,
                                                                                                                                                                            current_R2_SCORE,
                                                                                                                                                                            current_partial_learning_rate_for_session,
                                                                                                                                                                            model.batch_size))
                                    except tf.errors.OutOfRangeError:
                                        break
                            '''
                        except tf.errors.OutOfRangeError:
                            break
                num_steps_ = None
                if epoch_ == 0:
                    num_steps_ = params['initial_num_steps']
                else:
                    num_steps_ = params['num_steps']
                for i_train in range(num_steps_):
                    try:
                        start_time = time.time()
                        output_feed = [model.merged, model.MAE_, model.MSE_, model.MSLE_, model.R2_score_, model.row_indices, model.pm_values, model.batch_targets, model.predictions, model.learning_rate, model.train_op]
                        current_train_feed_dict = {}
                        for handle_idx_, current_handle_ in enumerate(model.list_of_handles):
                            current_train_feed_dict[current_handle_] = list_of_training_handles[handle_idx_]
                        current_train_feed_dict[model.learning_rate] = current_learning_rate
                        row_of_indices_for_batch_of_data = np.arange(model.batch_size)
                        np.random.shuffle(row_of_indices_for_batch_of_data)
                        current_train_feed_dict[model.row_indices] = row_of_indices_for_batch_of_data
                        current_train_feed_dict[model.is_training] = True
                        current_summary_train, current_MAE, current_MSE, current_MSLE, current_R2_SCORE, current_row_indices, current_pm_values, current_batch_targets, current_predictions, current_learning_rate_for_session, _ = sess.run(output_feed, feed_dict=current_train_feed_dict)
                        end_time = time.time()
                        print("TRUE PM VALUES")
                        for pm_val_ in current_pm_values:
                            print(pm_val_)
                        print("END OF TRUE PM VALUES")
                        print("PREDICTED PM VALUES")
                        for pm_predicted_val_ in current_predictions:
                            print(pm_predicted_val_)
                        print("END OF PREDICTED PM VALUES")
                        model.train_writer.add_summary(current_summary_train, model.global_step.eval())
                        print("(TRAINING-FULL) global step %d step-time %.2f MAE %.2f MSE %.2f MSLE %.2f R2 SCORE %.2f learning_rate %f batch_size %f" % (model.global_step.eval(),
                                                                                                                                                            end_time-start_time,
                                                                                                                                                            current_MAE,
                                                                                                                                                            current_MSE,
                                                                                                                                                            current_MSLE,
                                                                                                                                                            current_R2_SCORE,
                                                                                                                                                            current_learning_rate_for_session,
                                                                                                                                                            model.batch_size))
                        checkpoint_path_directory = params['logs_dir'] + "/" + params['experiment_dir']
                        checkpoint_path = os.path.join(checkpoint_path_directory, "translate.ckpt")
                        if i_train % 100 == 0 or i_train == (num_steps_-1):
                            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                        '''
                        if model.global_step.eval() % params['num_steps_before_validation_checkpoint'] == 0 and model.global_step.eval() != 0:
                            for i_eval_ in range(params['num_steps_for_validation']):
                                    try:
                                        output_feed = [model.merged, model.MAE_, model.MSE_, model.MSLE_, model.R2_score_, model.row_indices, model.learning_rate, model.global_eval_update_step_variable]
                                        current_eval_feed_dict = {}
                                        for handle_idx_, current_handle_ in enumerate(model.list_of_handles):
                                            current_eval_feed_dict[current_handle_] = list_of_eval_handles[handle_idx_]
                                        current_eval_feed_dict[model.learning_rate] = current_learning_rate
                                        row_of_indices_for_batch_of_data = np.arange(model.batch_size)
                                        np.random.shuffle(row_of_indices_for_batch_of_data)
                                        current_eval_feed_dict[model.row_indices] = row_of_indices_for_batch_of_data
                                        current_eval_feed_dict[model.is_training] = False
                                        current_summary_eval, current_MAE, current_MSE, current_MSLE, current_R2_SCORE, current_row_indices, current_learning_rate_for_session, _ = sess.run(output_feed, feed_dict=current_eval_feed_dict)
                                        eval_writer.add_summary(current_summary_eval, model.global_eval_step.eval())
                                        print("(EVALUATION - GLOBAL) GLOBAL STEP %d MAE %.2f MSE %.2f MSLE %.2f R2 SCORE %.2f learning_rate %f batch_size %f" % (model.global_eval_step.eval(),
                                                                                                                                                                    current_MAE,
                                                                                                                                                                    current_MSE,
                                                                                                                                                                    current_MSLE,
                                                                                                                                                                    current_R2_SCORE,
                                                                                                                                                                    current_learning_rate_for_session,
                                                                                                                                                                    model.batch_size))
                                    except tf.errors.OutOfRangeError:
                                        break
                        '''
                    except tf.errors.OutOfRangeError:
                        break

def run_evaluation_tfrecords(params, gpu_device, list_of_tfrecords_for_evaluation):
    with tf.Graph().as_default(), tf.device(gpu_device):
        with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:
            model = create_model(sess, params, list_of_trecords_for_evaluation=list_of_tfrecords_for_evaluation)

            eval_handle = sess.run(self.single_eval_iterator.string_handle())
            #for eval_iterator_ in model.list_of_eval_iterators:
            #    eval_handle = sess.run(eval_iterator_.string_handle())
            #    list_of_eval_handles.append(eval_handle)

            prefix_for_results_path = "/mnt/mnt/mounted_bucket" + "/results_for_final_presentation/" + params['experiment_dir']
            if not os.path.exists(prefix_for_results_path):
                os.makedirs(prefix_for_results_path)
            prefix_for_results_path = prefix_for_results_path +  "/" + params['type_of_evaluation']
            if not os.path.exists(prefix_for_results_path):
                os.makedirs(prefix_for_results_path)
            if params['pmRange'] != None:
                prefix_for_results_path = prefix_for_results_path + "/" + params['pmRange']
                if not os.path.exists(prefix_for_results_path):
                    os.makedirs(prefix_for_results_path)
            if params['year'] != None:
                prefix_for_results_path = prefix_for_results_path + "/" + params['year']
                if not os.path.exists(prefix_for_results_path):
                    os.makedirs(prefix_for_results_path)
            
            output_predicted_values = None
            output_predicted_values_npy_path_to_save = prefix_for_results_path + '/' + params['type_of_model'] + '_' + params['type_of_evaluation'] + '_' + params['webcamId'] + '_output_predicted_values'
            output_predicted_values_npy_path = output_predicted_values_npy_path_to_save + ".npy"
            output_true_values = None
            output_true_values_npy_path_to_save = prefix_for_results_path + '/' + params['type_of_model']  + '_' + params['type_of_evaluation'] + '_' + params['webcamId'] + '_output_true_values'
            output_true_values_npy_path = output_true_values_npy_path_to_save + ".npy"
            output_batch_inputs_of_highest_average_MAE = None
            output_predictions_of_highest_average_MAE = None
            output_targets_of_highest_average_MAE = None
            highest_average_MAE = None
            predictions_of_highest_average_MAE = None
            targets_of_highest_average_MAE = None
            batch_inputs_of_highest_average_MAE = None
            lowest_average_MAE = None
            predictions_of_lowest_average_MAE = None
            targets_of_lowest_average_MAE = None
            batch_inputs_of_lowest_average_MAE = None
            current_highest_average_MAE = None
            current_lowest_average_MAE = None

            
            if os.path.exists(output_predicted_values_npy_path):
                output_predicted_values = np.load(output_predicted_values_npy_path)
            else:
                output_predicted_values = []
                np.save(output_predicted_values_npy_path_to_save, output_predicted_values)

            if os.path.exists(output_true_values_npy_path):
                output_true_values = np.load(output_true_values_npy_path)
            else:
                output_true_values = []
                np.save(output_true_values_npy_path_to_save, output_true_values)

            for num_epochs in range(0, params['num_epochs']):
                for i_ in range(params['num_steps']):
                    try:
                        print("**********************************************")
                        print("Current Step %d ", i_)
                        start_time = time.time()
                        output_feed = [model.predictions, model.batch_targets, model.MAE_, model.R2_score_]
                        current_eval_feed_dict = {}
                        #for handle_idx_, current_handle_ in enumerate(model.list_of_handles):
                        #    current_eval_feed_dict[current_handle_] = list_of_eval_handles[handle_idx_]
                        current_eval_feed_dict[model.single_eval_handle] = eval_handle
                        row_of_indices_for_batch_of_data = np.arange(params['batch_size'])
                        np.random.shuffle(row_of_indices_for_batch_of_data)
                        current_eval_feed_dict[model.row_indices] = row_of_indices_for_batch_of_data
                        current_eval_feed_dict[model.is_training] = False
                        print("Output to Feed current information for current step")
                        output_predictions, output_targets, current_MAE, current_R2_SCORE = sess.run(output_feed, feed_dict=current_eval_feed_dict)
                        print("MAE %.2f R2 SCORE %.2f" % (current_MAE, current_R2_SCORE))
                        output_predicted_values.extend(output_predictions)
                        output_true_values.extend(output_targets)
                        np.save(output_predicted_values_npy_path_to_save, output_predicted_values)
                        np.save(output_true_values_npy_path_to_save, output_true_values)
                        end_time = time.time()
                        print("STEPTIME: %.2f" % (end_time - start_time))
                        print("***********************************************")
                    except tf.errors.OutOfRangeError:
                        break