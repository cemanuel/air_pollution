import os
import re
import datetime
import time
import pickle
import numpy as np
from skimage import filters
from skimage.feature import corner_harris, corner_peaks
from scipy import ndimage
from skimage import feature
from skimage.filters import threshold_otsu
from tensorflow.python.platform import gfile
import cv2

IMG_HEIGHT = 240
IMG_WIDTH = 320

def check_that_all_directories_have_same_experiment_folders(directories_to_search_in):
    """
        #################################################################################################################
        a.) Description:
        b.) Params:
        c.) Returns:
        d.) Raises:
        #################################################################################################################
    """
    experiment_directory_names = None
    for dir_to_search_in in directories_to_search_in:
        current_experiment_directory_names = []
        for f in os.listdir(dir_to_search_in):
            if re.match("\d{10}_experiment", f) != None:
                current_experiment_directory_names.append(f)
        if experiment_directory_names == None:
            experiment_directory_names = []
            experiment_directory_names.extend(current_experiment_directory_names)
        else:
            if set(experiment_directory_names) != set(current_experiment_directory_names):
                return False
    return True

def find_experiment_name(experiment_dir_name, experiment_dir_suffix, directories_to_search_in):
    """
        #################################################################################################################
        a) Description:
            1) pre-condition: 
                i) the prefix names of all experiment folders follow the following format:
                    * \d{10}_experiment_\d\d\d\d-\d\d-\d\d 
                    * \d{10} : refers to a unique value describing an experiment that corresponds when the experiment 
                        folder was created (e.g, '0000000001' will correspond to the first experiment folder that was
                        created)
                    * \d\d\d\d-\d\d-\d\d: referes to the time that the experiment folder was created 
                ii) the directories that we are searching in should contain the same number of experiment folders
            2) if experiment_dir_suffix is None, the name of the most recent experiment directory is returne
            3) if an experiment_dir is not None, we collect all names that match the following format:
                i) '\d{10}_experiment_\d\d\d\d-\d\d-\d\d' + experiment_dir_suffix
                ii) if there are multiple matches for a given directory, a message is printed to the console informing 
                    the user that the name provided was ambigious for that given directory and returns None
                iii) f there was no matches found for a given directory, a message is printed to the console informing 
                    the user that the name did not match any folder name in that given directory and returns None
                iv) if there was one single file that matches for a given directory, then this same file should be found
                    in the other directories listed in 'directories_to_search_in', where each of these directories should
                    contain only file that matches
        b) Params:
            1) params: dictionary of all parameters for a given experiment
        c) Returns:
            1) Prints a message on the status on finding an experiment folder the name provided in 
                params['experiment_dir']
            2) Returns a string that will be stored in params['experiment_dir'] or returns None if no name is found
        d.) Raises:
        #################################################################################################################
    """
    if not check_that_all_directories_have_same_experiment_folders(directories_to_search_in):
        print("All directories to be search in do not have the same experiment folders.")
        return None

    if experiment_dir_name == None and experiment_dir_suffix == None:
        experiment_directory_names = []

        for f in os.listdir(directories_to_search_in[0]):
            if re.match("\d{10}_experiment", f) != None:
                experiment_directory_names.append(f)

        if len(experiment_directory_names) > 0:
            most_recent_experiment_directory = sorted(experiment_directory_names)[len(experiment_directory_names)-1]
            return most_recent_experiment_directory
        else:
            print("There are no experiment directories in " + directories_to_search_in[0] + ".")
            return None
    else:
        matched_file = None
        for directory_to_search_in in directories_to_search_in:
            directories_that_match_pattern = []

            
            if experiment_dir_name != None:
                for f in os.listdir(directory_to_search_in):
                    if re.match(experiment_dir_name, f):
                        directories_that_match_pattern.append(f)
            else:
                for f in os.listdir(directory_to_search_in):
                    if re.match("\d{10}_experiment_\d\d\d\d-\d\d-\d\d_\d\d:\d\d:\d\d_" + experiment_dir_suffix + "$", f):
                        directories_that_match_pattern.append(f)

            if len(directories_that_match_pattern) == 0:
                print("The provided experiment directory name does not match any directory name in " + directory_to_search_in + ".")
                return

            if len(directories_that_match_pattern) > 1:
                print("The provided name of the experiment directory is ambigous in the " + directory_to_search_in + " directory.")
                print("Please use a more specific name.")
                return None

            if matched_file == None:
                matched_file = directories_that_match_pattern[0]
            else:
                if matched_file != directories_that_match_pattern[0]:
                    print("Two different folders matched the name provided in params['experiment_dir'] in two different directories.")
                    return None
        return matched_file 

def is_experiment_dir_string_unique_for_experiment_initialization(experiment_dir_prefix, experiment_dir_suffix, directories_to_search_in):
    """
        #################################################################################################################
        a) Description:
            1.) pre-Condition: 
                i) the prefix names of all experiment folders have the following format:
                    * \d{10}_experiment_\d\d\d\d-\d\d-\d\d 
                    * \d{10} : refers to a unique value describing an experiment that corresponds when the experiment 
                        folder was created (e.g, '0000000001' will correspond to the first experiment folder that was
                        created)
                    * \d\d\d\d-\d\d-\d\d: refers to the time that the experiment folder was created 
                ii) the directories that we are searching in should contain the same number of experiment folders
            2.) checks if prefix + 'experiment_dir_suffix' and 'experiment_dir_suffix' are unique directory name 
                across all directories listed in 'directories_to_search_in'
        b) Params:
            1.) experiment_dir_suffix:
            2.) directories_to_search_in:
        c) Returns:
            1.) boolean that determines whether a directory name is unique across all directories listed in 
                'directories_to_search_in'
        d) Raises:
        #################################################################################################################
    """
    if experiment_dir_suffix != None:
        for dir_to_search_in in directories_to_search_in:
            for f in os.listdir(dir_to_search_in):
                if re.match(experiment_dir_prefix, f) != None:
                    return False
                if experiment_dir_suffix != None:
                    if re.match("\d{10}_experiment_\d\d\d\d-\d\d-\d\d_\d\d:\d\d:\d\d_" + experiment_dir_suffix, f) != None:
                        return False
                    if re.match(experiment_dir_suffix, f) != None:
                        return False
    return True

def create_experiment_folders(experiment_dir_suffix, directories_to_create_experiment_folders_in):
    """
        #################################################################################################################
        a) Description:
            1.) Pre-Condition:
                i) the prefix names of all experiment folders have the following format:
                    * \d{10}_experiment_\d\d\d\d-\d\d-\d\d
                    * \d{10}: refers to a unique value describing the name of an experiment folder; the unique value 
                        corresponds to the order in which the experiment folders are created (e.g. '0000000001' will 
                        correspond to the first experiment folder that was created)
                    * \d\d\d\d-\d\d-\d\d: refers to the date and time that the experiment folder was created
            2.) creates an experiment directory in the directories listed in 'directories_to_search_in'
            3.) if experiment_dir_suffix is None:
                i) if 'experiment_dir_prefix' is unique across all directories listed in 
                    'directories_to_create_experiment_folders_in', then a folder with the name 'experiment_dir_prefix' is
                    created in each directory listed in 'directories_to_create_experiment_folders_in', 
                    'new_experiment_directory' is assigned 'experiment_dir_prefix', and 'new_experiment_directory' is 
                    returned
                ii) if 'experiment_dir_prefix' is not unique, then no experiment folders are created, 
                    'new_experiment_directory' is assigned None, and 'new_experiment_directory' is returned
            4.) if experiment_dir_suffix is not None:
                i) if 'experiment_dir_prefix' + '_' +  'experiment_dir_suffix' is unique across all directories listed in 
                    'directories_to_create_experiment_folders_in', then a folder with the name 
                    'experiment_dir_prefix' + '_' + 'experiment_dir_suffix' is created in each directory listed in 
                    'directories_to_create_experiment_folders_in'
                ii) if 'experiment_dir_prefix' + '_' +  'experiment_dir_suffix' is not unique, then no experiment folders 
                    are created, 'new_experiment_directory is assigned None', and 'new_experiment_directory' is returned
        b) Params:
            1.) experiment_dir_suffix:
            2.) directories_to_create_experiment_folders_in: ['data_dir', 'logs_dir', 'train_dir', 'dev_dir', 'test_dir', 'variables']
        c) Returns:
            1.) new_experiment_directory:
        d) Raises:
        #################################################################################################################
    """
    if not check_that_all_directories_have_same_experiment_folders(directories_to_create_experiment_folders_in):
        print("All directories to be search in do not have the same experiment folders.")
        return None

    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
    experiment_dir_prefix = None
    experiment_directory_names = []
    new_experiment_directory = None

    for f in os.listdir(directories_to_create_experiment_folders_in[0]):
        if re.match("\d{10}_experiment", f) != None:
            experiment_directory_names.append(f)

    if len(experiment_directory_names) > 0:
        most_recent_experiment_directory = sorted(experiment_directory_names)[len(experiment_directory_names)-1]
        experiment_number = int(most_recent_experiment_directory.split("_")[0]) + 1
        experiment_dir_prefix = str(experiment_number).zfill(10) + "_experiment_" + st
    else:
        experiment_dir_prefix = str(1).zfill(10) + "_experiment_" + st

    is_experiment_dir_unique = is_experiment_dir_string_unique_for_experiment_initialization(experiment_dir_prefix,
                                                                                                experiment_dir_suffix,
                                                                                                directories_to_create_experiment_folders_in)

    if not is_experiment_dir_unique:
        print("The name of the experiment directory that was provided is ambigious.") 
        print("Please provide a different experiment directory name.")
        return None

    if experiment_dir_suffix != None:
        new_experiment_directory = experiment_dir_prefix + "_" + experiment_dir_suffix
    else:
        new_experiment_directory = experiment_dir_prefix

    for dir_to_create_experiment_dir in directories_to_create_experiment_folders_in:
        os.makedirs(dir_to_create_experiment_dir + "/" + new_experiment_directory)

    return new_experiment_directory

def get_filenames_and_labels_of_images_from_path(path):
    raw_data_with_file_paths = np.load(path)
    shuffled_indices = np.arange(len(raw_data_with_file_paths))
    np.random.shuffle(shuffled_indices)
    raw_data_with_file_paths_shuffled = raw_data_with_file_paths[shuffled_indices]
    file_paths, webcamIds, webcamLats, webcamLongs, years, dates, hours, ranges_, pmValues = map(list, zip(*list(map(tuple, raw_data_with_file_paths_shuffled))))
    return file_paths, webcamLats, webcamLongs, webcamIds, years, dates, hours, ranges_, list(np.array(pmValues).astype(np.float32))

def prepare_data(data_path):
    """
        #################################################################################################################
        a.) Description:
        b.) Params:
        c.) Returns:
        d.) Raises:
        #################################################################################################################
    """
    return get_filenames_and_labels_of_images_from_path(data_path)


def interpolation_nearest_neighbor_helper(y, x, data_matrix):
    squared_distance_to_closest_point = float("inf")
    y_of_closest_non_zero_value_so_far = None
    x_of_closest_non_zero_value_so_far = None
    window_size = 3
    while(y_of_closest_non_zero_value_so_far == None and x_of_closest_non_zero_value_so_far == None):
        w = window_size // 2
        deltay = range(-1 * w, w+1)
        deltax = range(-1 * w, w+1)
        for dy in deltay:
            for dx in deltax:
                if y+dy < len(data_matrix) and x+dx < len(data_matrix[0]):
                    if data_matrix[y+dy][x+dx] > 0.001:
                        current_squared_distance = ((y+dy)**2) + ((x+dx)**2)
                        if (y_of_closest_non_zero_value_so_far == None and x_of_closest_non_zero_value_so_far == None):
                            squared_distance_to_closest_point = current_squared_distance
                            y_of_closest_non_zero_value_so_far = y+dy 
                            x_of_closest_non_zero_value_so_far = x+dx
                        else:
                            if current_squared_distance <  squared_distance_to_closest_point:
                                y_of_closest_non_zero_value_so_far = y+dy
                                x_of_closest_non_zero_value_so_far = x+dx
        window_size += 2
    return y_of_closest_non_zero_value_so_far, x_of_closest_non_zero_value_so_far
                        
def interpolation_nearest_neighbor(data_matrix):
    final_result = np.copy(data_matrix)
    for r in range(len(data_matrix)):
        for c in range(len(data_matrix[0])):
            if data_matrix[r][c] < 0.001:
                newR, newC = interpolation_nearest_neighbor_helper(r, c, data_matrix)
                final_result[r][c] = data_matrix[newR][newC]
    return final_result

def preprocess_images(params, filenames_of_images):
    processed_images = []
    for filename_of_image in filenames_of_images:
        print('/mnt' + filename_of_image)
        img = cv2.imread('/mnt' + filename_of_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processed_images.append(img)
    return processed_images
