from scipy.stats import spearmanr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from scipy.stats import rankdata
import time
from scipy.spatial.distance import pdist, squareform
import _pickle as cp

from IGTD.IGTD_FUNCTION import table_to_image, min_max_transform


def run(data_path, output_path):
    num_row = 32    # Number of pixel rows in image representation
    num_col = 24    # Number of pixel columns in image representation
    num = num_row * num_col # Number of features to be included for analysis, which is also the total number of pixels in image representation
    save_image_size = 3 # Size of pictures (in inches) saved during the execution of IGTD algorithm.
    max_step = 10000    # The maximum number of iterations to run the IGTD algorithm, if it does not converge.
    val_step = 300  # The number of iterations for determining algorithm convergence. If the error reduction rate
                    # is smaller than a pre-set threshold for val_step itertions, the algorithm converges.

    # Import the example data and linearly scale each feature so that its minimum and maximum values are 0 and 1, respectively.

    data = pd.read_csv(data_path, low_memory=False, sep=',', engine='c', na_values=['na', '-', ''],
                    header=0, index_col=0)
    data_code = data.iloc[:, num:num+num]
    data_nl = data.iloc[:, :num]
    norm_data_code = min_max_transform(data_code.values)
    norm_data_code = pd.DataFrame(norm_data_code, columns=data_code.columns, index=data_code.index)

    norm_data_nl = min_max_transform(data_nl.values)
    norm_data_nl = pd.DataFrame(norm_data_nl, columns=data_nl.columns, index=data_nl.index)


    # Run the IGTD algorithm using (1) the Euclidean distance for calculating pairwise feature distances and pariwise pixel
    # distances and (2) the absolute function for evaluating the difference between the feature distance ranking matrix and
    # the pixel distance ranking matrix. Save the result in Test_1 folder.
    fea_dist_method = 'Euclidean'
    image_dist_method = 'Euclidean'
    error = 'abs'

    result_dir = f'{output_path}/code'
    os.makedirs(name=result_dir, exist_ok=True)
    table_to_image(norm_data_code, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
                   max_step, val_step, result_dir, error)

    result_dir = f'{output_path}/nl'
    os.makedirs(name=result_dir, exist_ok=True)
    table_to_image(norm_data_nl, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
                   max_step, val_step, result_dir, error)


    # Run the IGTD algorithm using (1) the Pearson correlation coefficient for calculating pairwise feature distances,
    # (2) the Manhattan distance for calculating pariwise pixel distances, and (3) the square function for evaluating
    # the difference between the feature distance ranking matrix and the pixel distance ranking matrix.
    # Save the result in Test_2 folder.
    fea_dist_method = 'Pearson'
    image_dist_method = 'Manhattan'
    error = 'squared'
    result_dir = './Result/Train_2/code'
    os.makedirs(name=result_dir, exist_ok=True)
    table_to_image(norm_data_code, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
                   max_step, val_step, result_dir, error)

    result_dir = './Result/Train_2/nl'
    os.makedirs(name=result_dir, exist_ok=True)
    table_to_image(norm_data_nl, [num_row, num_col], fea_dist_method, image_dist_method, save_image_size,
                   max_step, val_step, result_dir, error)
