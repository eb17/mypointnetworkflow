"""
Purpose: Creation a dataset that is organized in the proper format and file structure for the experiments, as well as
expended by smote function. As result classe will be equal in number of points.
"""

import glob
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def load_folder_pts(path):
    """
    Load pts files. Each file has one class. The class name is translated into a code (number)
    :param path: dir and file of pts
    :type path: str
    :return: dataframe of point cloud (column = variables and row points)
    :rtype: pandas dataframe
    """
    all_files = glob.glob(path + "*.pts")
    print(all_files)
    try:
        list_of_dataframes = []
        list_of_names = []
        lengths_ = []

        i = 0
        for fp in all_files:
            df = pd.concat([pd.read_csv(fp, header=None,
                                        skiprows=1,
                                        delimiter=' ',
                                        usecols=[0, 1, 2],
                                        names=['x', 'y', 'z'])])
            df['GT'] = i
            i = i + 1
            lengths_.append(lengths(df))
            list_of_names.append(os.path.basename(fp).split('.')[0])
            list_of_dataframes.append(df)
        return list_of_names, list_of_dataframes, lengths_
    except:
        print(path + '.pts was not loaded!')


def create_dir(path):
    """
    Creating a sub folder of a given name
    :param path: path name
    :type path: str
    """
    access_rights = 0o755

    try:
        os.mkdir(path, access_rights)
        print("Successfully created the directory %s" % path)
    except OSError:
        print("Creation of the directory %s failed" % path)


if __name__ == '__main__':
    versuch = 'versuch_5-1'
    punktwolke = '3107'

    create_dir('Smote\\' + punktwolke)
    create_dir('Smote\\' + punktwolke + '\\' + versuch + '\\')

    list_names, list_df, list_len = load_folder_pts('Original\\' + punktwolke + '\\' + versuch + '\\')
    print(max(list_len))
    list_multiple = []

    for lengths in list_len:
        list_multiple.append(math.ceil(max(list_len) / lengths))

    print(list_multiple)
    print(list_len)
    print(list_names)

    for data, multiple, names in zip(list_df, list_multiple, list_names):
        if multiple == 1.0:
            print('the class ' + names + ' is the majority class')
            np.savetxt('Smote\\' + punktwolke + '\\' + versuch + '\\' + names + ".pts", data.to_numpy(), delimiter=" ")
        else:
            np_data = data.to_numpy()
            np.random.shuffle(np_data)

            sampled_data = np_data[:, :3]
            neighbor_points = NearestNeighbors(n_neighbors=int(multiple + 1), algorithm='ball_tree').fit(sampled_data)
            distances, indices = neighbor_points.kneighbors(sampled_data)

            new_points = sampled_data + (sampled_data - sampled_data[indices[:, 1]]) * random.uniform(0, 1)

            if int(multiple) > 2:
                for ii in range(2, int(multiple + 1)):
                    new_points = np.vstack([new_points, sampled_data + (
                            sampled_data - sampled_data[indices[:, ii]]) * random.uniform(0, 1)])

            print(names + ': ' + str(new_points.shape[0]))
            sampled_data = np.vstack([sampled_data, new_points])
            sampled_data = sampled_data[:max(list_len)]
            print(names + ': ' + str(sampled_data.shape[0]))

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            ax.scatter(sampled_data[:, 0], sampled_data[:, 1], sampled_data[:, 2], marker='^')
            ax.scatter(new_points[:, 0], new_points[:, 1], new_points[:, 2], marker='o')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')

            np.savetxt('Smote\\' + punktwolke + '\\' + versuch + '\\' + names + ".pts", sampled_data, delimiter=" ")
