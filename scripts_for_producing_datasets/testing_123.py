import numpy as np
import os
import re
from os.path import isfile
from datetime import datetime
from functools import reduce
import matplotlib.dates as mdates
import random
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--mounted_root_directory', type=str)
	parser.add_argument('--directory_to_save_dataset_by_webcamId_and_pmRange', type=str)
	parser.add_argument('--webcamIds_for_filtering', nargs=('*'))
	args = parser.parse_args()
	print(args.webcamIds_for_filtering)
	print(args.mounted_root_directory)