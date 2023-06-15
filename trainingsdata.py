"""
Purpose: Functions for loading data
"""

import glob

import pandas as pd
import tensorflow as tf


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
    class_num = list(range(len(all_files)))

    try:
        dataset = pd.concat(
            [pd.read_csv(fp, header=None, skiprows=1, delimiter=' ', usecols=[0, 1, 2], names=['x', 'y', 'z']).assign(
                GT=f) for f, fp in zip(class_num, all_files)])

        # Chose features : e.g.: 'sumE', 'ani', 'plan' or 10, 13, 14

        df['GT'] = df['GT'].astype(int)
        print('all pts files in folder' + path + ' successfully loaded! There are: ' + str(df.shape[0]) + ' points.')

        return dataset, class_num
    except ValueError:
        print(path + '.pts was not loaded!')


if __name__ == '__main__':
    df, num = load_folder_pts('Smote/0001/versuch_2-2/')

    # Data checking
    df = df.sample(frac=1)
    print(df.head())

    target = pd.get_dummies(df['GT'])
    df = df.drop(['GT'], axis=1)

    # Batch data check
    train_dataset = tf.data.Dataset.from_tensor_slices((df, target))
    train_dataset = train_dataset.shuffle(len(df)).batch(32)
    x, y = next(iter(train_dataset))
    print(x, y)
