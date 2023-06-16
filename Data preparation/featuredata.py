"""
Purpose: Functions to build data wit eigen value features
"""

import glob
import math
import os
from random import shuffle

import numpy as np
import open3d as o3d
import pandas as pd
from sklearn.neighbors import BallTree


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

    list_df = []
    list_names = []
    list_len = []
    count = 0

    try:
        for fp in all_files:
            df = pd.concat([pd.read_csv(fp, header=None,
                                        skiprows=1,
                                        delimiter=' ',
                                        usecols=[0, 1, 2],
                                        names=['x', 'y', 'z'])])
            df['GT'] = count
            count = count + 1
            list_len.append(len(df))
            list_names.append(os.path.basename(fp).split('.')[0])
            list_df.append(df)
        return list_names, list_df, list_len

    except ValueError:
        print(path + '.pts was not loaded!')


def create_dir(path):
    """
    Creating and sub folder of a given name
    :param path: path name
    :type path: str
    """
    access_rights = 0o755
    try:
        os.mkdir(path, access_rights)
        print("Successfully created the directory %s" % path)
    except OSError:
        print("Creation of the directory %s failed" % path)


def point_normals(data):
    """
    Calculate point normals
    :param data: data frame of point cloud
    :type data: npy-array
    :return: data frame of point cloud with added feature normals
    :rtype: npy-array
    """
    # creat an open3d point cloud object
    pcd = o3d.geometry.PointCloud()
    # fill the object with my data
    pcd.points = o3d.utility.Vector3dVector(data[:, :3])
    # calculate the point normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
    # open3d point normals to npy-array
    ds_normals = np.asarray(pcd.normals)
    # add point normals as variable to data set
    ds = np.c_[data, ds_normals]
    return ds


def eig_non_zero(eig_v):
    sum_e = sum(eig_v)
    for points in range(0, 3):
        eig_v[points] = eig_v[points] / sum_e
        if eig_v[points] <= 0:
            eig_v[points] = 0.000000001
    return eig_v


def dataset(data, point_cloud_name):
    """
    Created
    :param data: loaded original data
    :type data: npy
    :param point_cloud_name:  name of the current point cloud
    :type point_cloud_name: str
    """

    new_data_array = np.array([]).reshape([0, 30])
    for t in range(0, 1):

        data = point_normals(data)

        tree = BallTree(data[:, :3], leaf_size=100)
        max_distance = 0.035
        ind, dist = tree.query_radius(data[:, :3], r=max_distance, return_distance=True)

        # Declare list for variables
        l_eig = []
        l_sum = []
        l_omn = []
        l_ent = []
        l_ani = []
        l_plan = []
        l_lin = []
        l_sur = []
        l_sph = []
        l_edg = []
        l_mom11 = []
        l_mom21 = []
        l_mom12 = []
        l_mom22 = []
        l_density = []
        l_ver = []

        print('Starting with feature calculation')

        ii = 0
        for inde, distance in zip(ind, dist):

            number_of_points = inde.tolist()
            if len(number_of_points) > 100:
                shuffle(number_of_points)
                number_of_points = number_of_points[:100]
                ii = ii + 1

            new_xyz = data[number_of_points, :3]
            new_zn = data[number_of_points, :-1]

            # calculate CoVariance matrix
            if len(new_xyz) < 2:
                new_xyz = np.r_[new_xyz, new_xyz]

            cov_xyz = np.cov(new_xyz, rowvar=0)

            # calculate eigenvalues and eigenvectors
            eigenvalue, eigen_vector = np.linalg.eig(cov_xyz)

            # sort the eigenvalues
            eigenvalue = np.sort(eigenvalue)
            eigen_vector = np.sort(eigen_vector)

            l_sum.append(eigenvalue[2] + eigenvalue[1] + eigenvalue[0])
            # check if eigenvalue is non zero and sum eigenvalues to 1
            eigenvalue = eig_non_zero(eigenvalue)

            l_eig.append(eigenvalue)
            l_density.append(len(number_of_points) / ((4 / 3) * math.pi * math.pow(max_distance, 3)))
            l_omn.append(np.power(eigenvalue[2] * eigenvalue[1] * eigenvalue[0], 1 / 3))
            l_ent.append(np.sum(eigenvalue * np.log(eigenvalue)) * -1)
            l_ani.append((eigenvalue[2] - eigenvalue[0]) / eigenvalue[2])
            l_plan.append((eigenvalue[1] - eigenvalue[0]) / eigenvalue[2])
            l_lin.append((eigenvalue[2] - eigenvalue[1]) / eigenvalue[2])
            l_sur.append(eigenvalue[0] / (eigenvalue[2] + eigenvalue[1] + eigenvalue[0]))
            l_sph.append(eigenvalue[0] / eigenvalue[2])
            l_edg.append(eigenvalue[0] / (eigenvalue[2] - eigenvalue[0]))
            l_mom11.append(sum(np.dot((new_xyz[1:] - new_xyz[0]), eigen_vector[0])))
            l_mom12.append(sum((np.dot(new_xyz[1:] - new_xyz[0], eigen_vector[0]))) ** 2)
            l_mom21.append(sum(np.dot((new_xyz[1:] - new_xyz[0]), eigen_vector[1])))
            l_mom22.append(sum((np.dot(new_xyz[1:] - new_xyz[0], eigen_vector[1]))) ** 2)
            l_ver.append(1 - new_zn[0])
        data = np.c_[data, np.array(l_eig).real]
        data = np.c_[data, np.array(l_sum).real]
        data = np.c_[data, np.array(l_omn).real]
        data = np.c_[data, np.array(l_ent).real]
        data = np.c_[data, np.array(l_ani).real]
        data = np.c_[data, np.array(l_plan).real]
        data = np.c_[data, np.array(l_lin).real]
        data = np.c_[data, np.array(l_sur).real]
        data = np.c_[data, np.array(l_sph).real]
        data = np.c_[data, np.array(l_edg).real]
        data = np.c_[data, np.array(l_mom11).real]
        data = np.c_[data, np.array(l_mom12).real]
        data = np.c_[data, np.array(l_mom21).real]
        data = np.c_[data, np.array(l_mom22).real]
        data = np.c_[data, np.array(l_density).real]
        data = np.c_[data, np.array(l_ver).real]
        new_data_array = np.r_[new_data_array, data]

        x = new_data_array[:, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]]
        header = "x, y, z, c, nx, ny, nz, e1, e2, e3, sumE, omn, ent, ani, plan, lin,sur, sph, edg, mo11, mo21, mo21, mo22, dic, ver"
        np.savetxt(str(point_cloud_name) + '.pts', x, delimiter=',', header=header)
        return x


if __name__ == '__main__':
    versuch = 'versuch_2-2'
    punktwolke = '0004'

    create_dir('Feature/' + punktwolke)
    create_dir('Feature/' + punktwolke + '/' + versuch + '/')

    list_names, list_df, list_len = load_folder_pts('Original/' + punktwolke + '/' + versuch + '/')
    ll = np.array([]).reshape(0, 4)

    for l_df in list_df:
        ll = np.r_[ll, l_df]

    alldata = dataset(ll, punktwolke)

    for i, name in enumerate(list_names):
        print(name, i)
        all_temp = alldata[alldata[:, 3] == i]
        all_temp = all_temp[:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]
        np.savetxt('Feature/' + punktwolke + '/' + versuch + '/' + name + ".pts", all_temp, delimiter=" ")
