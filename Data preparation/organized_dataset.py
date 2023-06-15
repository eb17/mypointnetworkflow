"""
Purpose: Creation a dataset that is organized in the proper format and file structure for the experiments
"""

import pandas as pd
import glob
import os
import numpy as np


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
            dataset = pd.concat([pd.read_csv(fp, header=None,
                                             skiprows=1,
                                             delimiter=' ',
                                             usecols=[0, 1, 2],
                                             names=['x', 'y', 'z'])])
            dataset['GT'] = i
            i = i + 1
            lengths_.append(lengths_(dataset))
            list_of_names.append(os.path.basename(fp).split('.')[0])
            list_of_dataframes.append(dataset)
        return list_of_names, list_of_dataframes, lengths_
    except:
        print(path + '.pts was not loaded!')


def create_dir(path):
    """
    Creating a sub folder of a given name
    :param path: path name
    """
    access_rights = 0o755

    try:
        os.mkdir(path, access_rights)
        print("Successfully created the directory %s" % path)
    except OSError:
        print("Creation of the directory %s failed" % path)


if __name__ == '__main__':
    merge1 = "Tisch"
    merge2 = "Stuhl"
    new_class = "Bestuhlung"
    versuch = 'versuch_4-2'
    versuch_c = 'versuch_4-3'
    punktwolke = '0010'
    typ = 'Original\\'
    create_dir(typ + punktwolke)
    create_dir(typ + punktwolke + '\\' + versuch_c + '\\')

    list_names, list_df, list_len = load_folder_pts('Original\\' + punktwolke + '\\' + versuch + '\\')
    print(list_names)

    print('\n')
    for class_name, lengths in zip(list_names, list_len):
        print(lengths)

    new_df = np.array([]).reshape(0, list_df[0].shape[1])

    for class_name, df in zip(list_names, list_df):
        if class_name == merge1:
            new_df = np.r_[new_df, df]
            np.savetxt(typ + punktwolke + '\\' + versuch_c + '\\' + class_name + ".pts", df, delimiter=" ")
        elif class_name == merge2:
            new_df = np.r_[new_df, df]
            np.savetxt(typ + punktwolke + '\\' + versuch_c + '\\' + class_name + ".pts", df, delimiter=" ")
        else:
            continue
