import math
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import tensorflow as tf
from pyntcloud import PyntCloud
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from tensorflow import keras

import pointnet_keras as cnn
import trainingsdata as td
import weighted_categorical_crossentropy as wcc


def run_experiment(epochs, filepath, loss, use_trained, load_weights):
    """
    Execution of the training. Callbacks: EarlyStopping und ModelCheckpoint (Save only best model).
    :param epochs: Number of epochs
    :param filepath: Directory for saving weights
    :param loss: Typ of loss function
    :param use_trained: boolean parameter to chose pre-trained weights
    :param load_weights: Directory where the weights loaded form
    :return: model und logging
    """
    model = cnn.get_shape_segmentation_model_org(NUM_SAMPLE_POINTS, NUM_CLASSES)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=loss,
        metrics=["accuracy"],
    )

    checkpoint_filepath = filepath + "checkpoint"
    keras_callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, mode='min', min_delta=0.01),
        keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_loss', save_best_only=True,
                                        save_weights_only=True, mode='min')]

    if use_trained == True:
        model.load_weights(load_weights + "checkpoint")

    history_training = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=keras_callbacks,
    )

    model.load_weights(checkpoint_filepath)
    return model, history_training


def plot_result(item, directory, name):
    """
    Creat plot of metrics from the training history
    :param item: history mit allen metrics ACC und LOSS
    :param directory: Directory to store the result files of this experiment
    :param name: filename
    """
    fig = plt.figure()
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    fig.savefig(directory + name + '.png')


def helmert_transformation(start_points, r, m, t_m):
    """
    3D helmert transformation
    :param start_points: Points to transform
    :param r: Rotation matrix
    :param m: Scale parameter
    :param t_m: Translation matrix
    :return target_points: Transformed points
    """
    target_points = np.dot(r, start_points.t_m) * m
    target_points = target_points.t_m + t_m.T
    return target_points


def point_normals(dataset_without_normals):
    """
    Calculation of the 3D-point normals
    :param dataset_without_normals: data frame of point cloud
    :return dataset_normals: data frame of point cloud with added feature normals
    """
    # creat an open3d point cloud object
    pcd = o3d.geometry.PointCloud()
    # fill the object with my data
    pcd.points = o3d.utility.Vector3dVector(dataset_without_normals[:, :3])
    # calculate the point normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=40))
    # open3d point normals to npy-array
    ds_normals = np.asarray(pcd.normals)
    # add point normals as variable to data set
    dataset_normals = np.c_[dataset_without_normals, ds_normals]
    return dataset_normals


def global_norm(points_orig):
    """
    The coordinates and feature values are transferred into the range  from 0 to 1. Coordinates vary very strongly
    according to the point cloud
    :param points_orig: Original scaled coordinates and features
    :return points_norm: Normalized coordinates and features
    """
    points_norm = np.c_[
        points_orig, (points_orig[:, 0] - min(points_orig[:, 0])) / (max(points_orig[:, 0]) - min(points_orig[:, 0]))]
    points_norm = np.c_[
        points_norm, (points_norm[:, 1] - min(points_norm[:, 1])) / (max(points_norm[:, 1]) - min(points_norm[:, 1]))]
    points_norm = np.c_[
        points_norm, (points_norm[:, 2] - min(points_norm[:, 2])) / (max(points_norm[:, 2]) - min(points_norm[:, 2]))]
    points_norm[:, 3] = (points_norm[:, 3] - min(points_norm[:, 3])) / (max(points_norm[:, 3]) - min(points_norm[:, 3]))
    points_norm[:, 4] = (points_norm[:, 4] - min(points_norm[:, 4])) / (max(points_norm[:, 4]) - min(points_norm[:, 4]))
    points_norm[:, 5] = (points_norm[:, 5] - min(points_norm[:, 5])) / (max(points_norm[:, 5]) - min(points_norm[:, 5]))
    return points_norm


def data_prep(point_cloud_data, voxel_size, voxel_size_h, training):
    """
    Preprocessing of points, since most networks have a fixed input of points. This is controlled by the voxel structure
     (sub point clouds)
    :param point_cloud_data: Point cloud only with coordinates (x, y, z)
    :param voxel_size: Horizontal size of voxel (sup-point cloud)
    :param voxel_size_h: Vertical size of voxel (sup-point cloud)
    :param training: Boolean operator for training and test data
    :return points_train, labels_train, num_classes: Datasets divided into features, labels und statistical value no. of
     classes
    """

    point_cloud_data = point_cloud_data.replace(np.nan, 0)
    data = point_cloud_data.sample(frac=1, random_state=1)

    data[['x', 'y']] = data[['x', 'y']] - data[['x', 'y']].median()
    index_names = data[data['z'] < -2].index  # Max resolution of LIDAR system
    data.drop(index_names, inplace=True)
    data['z'] = data['z'] - data['z'].min()

    # separate input and target data
    target = pd.get_dummies(data['GT'])
    data = data.drop(['GT'], axis=1)

    df_1 = data.iloc[:len(data) // 3]
    target_1 = target.iloc[:len(data) // 3]

    df_2 = data.iloc[len(data) // 3:(len(data) // 3) * 2]
    target_2 = target.iloc[len(data) // 3:(len(data) // 3) * 2]

    df_3 = data.iloc[(len(data) // 3) * 2:]
    target_3 = target.iloc[(len(data) // 3) * 2:]

    df_2.x[df_2['x'] == df_2['x'].min()] = df_2['x'].min() - voxel_size / 2
    df_2.y[df_2['y'] == df_2['y'].min()] = df_2['y'].min() - voxel_size / 2

    df_3.x[df_3['x'] == df_3['x'].min()] = df_3['x'].min() - voxel_size / 3
    df_3.y[df_3['y'] == df_3['y'].min()] = df_3['y'].min() - voxel_size / 3

    df_1.reset_index(drop=True, inplace=True)
    target_1.reset_index(drop=True, inplace=True)
    df_2.reset_index(drop=True, inplace=True)
    target_2.reset_index(drop=True, inplace=True)
    df_3.reset_index(drop=True, inplace=True)
    target_3.reset_index(drop=True, inplace=True)

    m = 1
    t_array = np.array([0, 0, 0])
    if training == True:
        #  The training data is rotated six times

        p1 = df_1.to_numpy()
        p2 = df_2.to_numpy()
        p3 = df_3.to_numpy()

        p11 = df_1.to_numpy()
        p12 = df_2.to_numpy()
        p13 = df_3.to_numpy()

        p21 = df_1.to_numpy()
        p22 = df_2.to_numpy()
        p23 = df_3.to_numpy()

        r2 = np.array(
            [[math.cos(math.radians(90)), -math.sin(math.radians(90)), 0],
             [math.sin(math.radians(90)), math.cos(math.radians(90)), 0], [0, 0, 1]])
        p1[:, :3] = helmert_transformation(p1[:, :3], r2, m, t_array)
        p2[:, :3] = helmert_transformation(p2[:, :3], r2, m, t_array)
        p3[:, :3] = helmert_transformation(p3[:, :3], r2, m, t_array)

        r1 = np.array(
            [[math.cos(math.radians(180)), -math.sin(math.radians(180)), 0],
             [math.sin(math.radians(180)), math.cos(math.radians(180)), 0], [0, 0, 1]])

        p11[:, :3] = helmert_transformation(p11[:, :3], r1, m, t_array)
        p12[:, :3] = helmert_transformation(p12[:, :3], r1, m, t_array)
        p13[:, :3] = helmert_transformation(p13[:, :3], r1, m, t_array)

        p1 = global_norm(p1)
        p2 = global_norm(p2)
        p3 = global_norm(p3)

        p11 = global_norm(p11)
        p12 = global_norm(p12)
        p13 = global_norm(p13)

        p21 = global_norm(p21)
        p22 = global_norm(p22)
        p23 = global_norm(p23)

        p1 = point_normals(p1)
        p2 = point_normals(p2)
        p3 = point_normals(p3)

        p11 = point_normals(p11)
        p12 = point_normals(p12)
        p13 = point_normals(p13)

        p21 = point_normals(p21)
        p22 = point_normals(p22)
        p23 = point_normals(p23)

        # berechnen der Normalen
        points_ = [p21, p22, p23, p1, p2, p3, p11, p12, p13]
        # points_ = [p21,  p1, p11]
        labels_ = [target_1.to_numpy(), target_2.to_numpy(), target_3.to_numpy(), target_1.to_numpy(),
                   target_2.to_numpy(), target_3.to_numpy(), target_1.to_numpy(), target_2.to_numpy(),
                   target_3.to_numpy()]
        # labels_ = [target.to_numpy(), target.to_numpy(), target.to_numpy()]
    else:
        # Für das Testen werden nur die originalen Daten verwendet.
        p1 = df_1.to_numpy()
        p2 = df_2.to_numpy()
        p3 = df_3.to_numpy()

        p1 = global_norm(p1)
        p2 = global_norm(p2)
        p3 = global_norm(p3)

        p1 = point_normals(p1)
        p2 = point_normals(p2)
        p3 = point_normals(p3)

        points_ = [p1, p2, p3]
        labels_ = [target_1.to_numpy(), target_2.to_numpy(), target_3.to_numpy()]

    num_classes = target.to_numpy().shape[1]

    points_train = np.array([]).reshape([0, NUM_FEATURE + 3])  #
    labels_train = np.array([]).reshape([0, num_classes])

    # Organize in voxel structure
    for points, labels in zip(points_, labels_):

        # Convert pandas dataframe to pyntcloud
        df = pd.DataFrame(points[:, :3], columns=['x', 'y', 'z'])
        cloud = PyntCloud(df)
        # Create voxels
        voxel_grid_id = cloud.add_structure("voxelgrid", size_x=voxel_size, size_y=voxel_size, size_z=voxel_size_h)
        voxel_grid = cloud.structures[voxel_grid_id]
        voxels = np.unique(voxel_grid.voxel_n)

        i = 0
        for voxel in voxels:
            """
            Voxel by voxel, the points are loaded based on the indexes. Only voxels that have the min. number of points 
            are used. In order to create a whole number, points are duplicated in case of lack. Points are scaled to
            a fixed range of values (voxel size) from 0 to 1. The height to the maximum room height. Then the points 
            are divided into training and evaluation data.
            """

            index = np.where(voxel_grid.voxel_n == voxel)[0].tolist()
            if NUM_SAMPLE_POINTS * .5 <= len(index):

                points_out = points[index, :3] - voxel_grid.voxel_centers[voxel]
                points_out = np.c_[points_out, points[index, 3:6]]
                points_out = np.c_[points_out, points[index, 6:12]]  # Option: Change 9 : 12
                points_out = np.c_[points_out, np.zeros([len(points_out), 3])]
                points_out[:, 12:15] = voxel_grid.voxel_centers[voxel]  # Option: Change 9 : 12 and 12 : 15
                labels_out = labels[index, :num_classes]

                points_out[:, :2] = (points_out[:, :2] - (-voxel_size)) / (voxel_size - (-voxel_size))
                points_out[:, 2] = (points_out[:, 2] - (-voxel_size_h)) / (voxel_size_h - (-voxel_size_h))
                add_points = (len(points_out) // NUM_SAMPLE_POINTS) * NUM_SAMPLE_POINTS + NUM_SAMPLE_POINTS - len(
                    points_out)
                points_out = np.r_[points_out, points_out[:add_points]]
                labels_out = np.r_[labels_out, labels_out[:add_points]]

                # Idea source: https://stackoverflow.com/questions/4601373/

                points_out, labels_out = shuffle(points_out, labels_out, random_state=0)
                points_train = np.r_[points_train, points_out]
                labels_train = np.r_[labels_train, labels_out]

            else:
                continue
            i = i + 1

    return points_train, labels_train, num_classes


def bba(train_points_r, train_labels_r, num_classes, train_labels):
    print('----------------------')
    list_class_count = np.count_nonzero(train_labels, axis=0)
    all_training_examples = sum(list_class_count)
    list_ppc = []
    list_ppca = []

    for class_count in list_class_count:
        list_ppca.append(all_training_examples / num_classes - class_count)
        list_ppc.append(1 / num_classes - class_count / all_training_examples)

    item_index = [index for index, value in enumerate(list_ppc) if value < 0]
    points_per_batch = train_labels_r.sum(axis=1) / train_labels_r.shape[1]
    augmented_points = np.array([]).reshape(-1, NUM_SAMPLE_POINTS, NUM_FEATURE)
    augmented_labels = np.array([]).reshape(-1, NUM_SAMPLE_POINTS, NUM_CLASSES)

    for train_point, train_label, points_per_batch_ in zip(train_points_r, train_labels_r, points_per_batch):
        if all(v != np.argmax(points_per_batch_, axis=0) for v in item_index):
            augmented_points = np.r_[augmented_points, train_point.reshape(-1, NUM_SAMPLE_POINTS, NUM_FEATURE)]
            augmented_labels = np.r_[augmented_labels, train_label.reshape(-1, NUM_SAMPLE_POINTS, NUM_CLASSES)]
        else:
            continue

    # Idea source: https://stackoverflow.com/questions/405516/
    # Idea source: https://stackoverflow.com/questions/7270321/

    portion = augmented_points.shape[0] / train_points_r.shape[0]
    reps = 1

    while portion < (1 / NUM_CLASSES):
        portion = ((augmented_points.shape[0] * reps) / (train_points_r.shape[0] + augmented_points.shape[0] * reps))
        reps += 1
    print('the small class is repeated ', reps - 1)

    if reps < 500:
        augmented_points = np.repeat(augmented_points, repeats=reps - 1, axis=0)
        augmented_labels = np.repeat(augmented_labels, repeats=reps - 1, axis=0)

    # Idea source: https://stackoverflow.com/questions/53239242/how-to-duplicate-each-row-of-a-matrix-n-times-numpy
    train_points_r = np.r_[train_points_r, augmented_points]
    train_labels_r = np.r_[train_labels_r, augmented_labels]
    train_points_r, train_labels_r = shuffle(train_points_r, train_labels_r, random_state=0)

    return train_points_r, train_labels_r


def training_eval_data(list_eval_data, voxel_size_test, voxel_size_test_h, experiment_name_eval):
    """
    Data pre-processing in form of a batch that can be given in to the network
    :param list_eval_data: List of sub-point clouds
    :param voxel_size_test: Horizontal size of voxel (sup-point cloud)
    :param voxel_size_test_h: Vertical size of voxel (sup-point cloud)
    :param experiment_name_eval: Name of experiment
    :return training_dataset, valuation_dataset, all_training_examples, class_weights_a, class_weights_b: Dataset and
    batches for the training and online evaluation, as well as weights for the weighted loss functions
    """
    points_trains = np.array([]).reshape([0, NUM_FEATURE])
    labels_trains = np.array([]).reshape([0, NUM_CLASSES])

    for df_entry in list_eval_data:
        points_train_temp, labels_train_temp, num_classes = data_prep(df_entry, voxel_size_test, voxel_size_test_h,
                                                                      True)
        points_trains = np.r_[points_trains, points_train_temp[:, :NUM_FEATURE]]
        labels_trains = np.r_[labels_trains, labels_train_temp]

    # Creation of training data
    train_points_r = points_trains.reshape(-1, NUM_SAMPLE_POINTS, NUM_FEATURE)
    train_labels_r = labels_trains.reshape(-1, NUM_SAMPLE_POINTS, NUM_CLASSES)

    # Batch optimization: on / off
    if BOP == True:
        train_points_r, train_labels_r = bba(train_points_r, train_labels_r, num_classes, labels_trains)

    p = int((train_points_r.shape[0] * 0.8))
    p2 = int(train_points_r.shape[0] - train_points_r.shape[0] * 0.8)

    tp = train_points_r[:p]  # Training points
    ep = train_points_r[:p2]  # Evaluation points
    tl = train_labels_r[:p]  # Training labels
    el = train_labels_r[:p2]  # Evaluation labels

    training_dataset = tf.data.Dataset.from_tensor_slices((tp, tl))
    training_dataset = training_dataset.shuffle(len(training_dataset)).batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.from_tensor_slices((ep, el))
    valuation_dataset = test_dataset.shuffle(len(test_dataset)).batch(BATCH_SIZE)

    # Point cloud statistics
    result_file_name_stat = 'Versuch/' + experiment_name_eval + '/'

    train_labels = tl.reshape(tl.shape[0] * tl.shape[1], tl.shape[2])
    list_class_numbers = range(0, num_classes)

    print('################# Information about training data #################### \n')
    list_class_count = []
    for class_number in list_class_numbers:
        list_class_count.append(np.sum(train_labels[:, class_number] == 1))
    all_training_examples = sum(list_class_count)

    print('No. of training points', str(all_training_examples))
    with open(result_file_name_stat + 'stat.csv', "a") as func:
        func.write(
            '################# Information about training data #################### \n No. of training points: ;' + str(
                all_training_examples) + '\n')
    class_weights_a = []
    class_weights_b = []
    for class_count, class_number in zip(list_class_count, list_class_numbers):
        anteil = class_count / sum(list_class_count)
        soll_anteil = 1 / len(list_class_numbers)
        class_weights_a.append(1 - anteil)
        class_weights_b.append(soll_anteil - anteil + 1)  # Berechnung des Anteils für gewichtete Loss Funktion.
        print('CLASS PORTION', str(class_number), ': ', anteil)
        with open(result_file_name_stat + 'stat.csv', "a") as func:
            func.write('CLASS PORTION' + str(class_number) + ': ;' + str(anteil) + '\n')

    print('################# Information about evaluation data #################### \n')

    eval_labels = el.reshape(el.shape[0] * el.shape[1], el.shape[2])
    list_class_count = []
    for class_number in list_class_numbers:
        list_class_count.append(np.sum(eval_labels[:, class_number] == 1))

    print('No. of evaluation points', str(sum(list_class_count)))
    with open(result_file_name_stat + 'stat.csv', "a") as func:
        func.write(
            '########### Information about evaluation data ############# \n No. of evaluation points: ;' + str(
                sum(list_class_count)) + '\n')

    for class_count, class_number in zip(list_class_count, list_class_numbers):
        print('CLASS PORTION', str(class_number), ': ', class_count / sum(list_class_count))
        with open(result_file_name_stat + 'stat.csv', "a") as func:
            func.write('CLASS PORTION' + str(class_number) + ': ;' + str(class_count / sum(list_class_count)) + '\n')

    return training_dataset, valuation_dataset, all_training_examples, class_weights_a, class_weights_b


def testing_data(list_test_data, voxel_size_test, voxel_size_test_h, experiment_name_test):
    """
    Data pre-processing in form of a batch that can be given in to the network for testing
    :param list_test_data: List of loaded sub point clouds (e.g. spitted by room names)
    :param voxel_size_test: Horizontal size of voxel (sup-point cloud)
    :param voxel_size_test_h: Vertical size of voxel (sup-point cloud)
    :param experiment_name_test: Name of experiment
    :return: dataset_test, test_points: Dataset for test with unknown data and dataset only with coordinates
    """
    test_points, test_labels, num_classes = data_prep(list_test_data, voxel_size_test, voxel_size_test_h, False)
    result_dir = 'Versuch/' + experiment_name_test + '/'
    test_points_b = test_points[:, :NUM_FEATURE]

    # Creat test data
    points_r = test_points_b.reshape(-1, NUM_SAMPLE_POINTS, NUM_FEATURE)
    labels_r = test_labels.reshape(-1, NUM_SAMPLE_POINTS, NUM_CLASSES)

    dataset_test = tf.data.Dataset.from_tensor_slices((points_r, labels_r))
    dataset_test = dataset_test.batch(BATCH_SIZE)

    # Point cloud statistics
    list_class_numbers = range(0, num_classes)
    print('################# Information about TESTING data #################### \n')
    list_class_count = []

    for class_number in list_class_numbers:
        list_class_count.append(np.sum(test_labels[:, class_number] == 1))
    all_training_examples = sum(list_class_count)
    print('No. of testing points', str(all_training_examples))

    with open(result_dir + 'stat.csv', "a") as func:
        func.write(
            '################# Information about TESTING data #################### \n No. of testing points: ;' + str(
                all_training_examples) + '\n')

    for class_count, class_number in zip(list_class_count, list_class_numbers):
        print('CLASS PORTION', str(class_number), ': ', class_count / sum(list_class_count))
        with open(result_dir + 'stat.csv', "a") as func:
            func.write('CLASS PORTION' + str(class_number) + ': ;' + str(class_count / sum(list_class_count)) + '\n')

    return dataset_test, test_points


def create_dir(path):
    """
    Creating a sub folders for each experiment
    :param path: path name
    """
    access_rights = 0o755

    try:
        os.mkdir(path, access_rights)
        print("Successfully created the directory %s" % path)
    except OSError:
        print("Creation of the directory %s failed" % path)


def prediction(test_dataset, filename_test_data, number_of_classes, dir_results, number_of_sample_pts, voxel_size_test,
               voxel_size_test_h, room_number, coordinates_test_data):
    """
    Prediction of semantic classes by means of the created algorithm
    :param coordinates_test_data: Original coordinates of points to create semantic point cloud
    :param room_number: Number or name of the room in the dataset
    :param test_dataset: Batch/ Dataset of unknown data for independent tests
    :param filename_test_data: Name of point cloud file
    :param number_of_classes: Number of classes in the test dataset
    :param dir_results: Directory to store results
    :param number_of_sample_pts: Number of points as network input
    :param voxel_size_test: Horizontal size of voxel (sup-point cloud)
    :param voxel_size_test_h: Vertical size of voxel (sup-point cloud)
    """
    gt = np.empty(shape=0)
    coordinates = np.array([]).reshape(0, NUM_FEATURE + 3)
    pred = np.empty(shape=0)
    vv = 0
    ww = 1

    for element in test_dataset:
        val_predictions = segmentation_model.predict(element[0])
        pred = np.r_[pred, val_predictions.argmax(axis=2).reshape(len(val_predictions) * number_of_sample_pts)]
        gt = np.r_[gt, np.array(element[1]).argmax(axis=2).reshape(len(element[1]) * number_of_sample_pts)]
        coordinates = np.r_[
            coordinates, coordinates_test_data[vv * len(val_predictions) * number_of_sample_pts:ww * len(
                val_predictions) * number_of_sample_pts]]
        vv = vv + 1
        ww = ww + 1

    coordinates[:, :2] = (coordinates[:, :2] * (voxel_size_test - (-voxel_size_test)) + (-voxel_size_test))
    coordinates[:, 2] = (coordinates[:, 2] * (voxel_size_test_h - (-voxel_size_test_h)) + (-voxel_size_test_h))
    coordinates[:, :3] = coordinates[:, :3] + coordinates[:, NUM_FEATURE:]

    pred_pc = np.c_[coordinates[:, :3], pred]

    # Point-wise storing by class
    for uu in range(0, number_of_classes):
        pred_pc_temp = pred_pc[pred_pc[:, 3] == uu]
        np.savetxt(pc_dir + room_number[:-1] + '-' + str(filename_test_data) + '-' + str(uu) + "pc.pts", pred_pc_temp,
                   delimiter=" ")

    # Creat a confusion matrix and calculate precision and recall
    cm = confusion_matrix(gt, pred)
    pp_rc = np.zeros((number_of_classes, 2))

    for c in range(0, number_of_classes):
        tp = cm[c, c]
        if cm[c].sum() == 0:
            sum_pre = 0.000001
        else:
            sum_pre = cm[c].sum()
        if cm[:, c].sum() == 0:
            sum_org = 0.000001
        else:
            sum_org = cm[:, c].sum()
        pp_rc[c, 0] = (tp / sum_org) * 100
        pp_rc[c, 1] = (tp / sum_pre) * 100
    result = np.c_[cm, pp_rc]

    with open(dir_results + 'results.csv', "a") as func:
        func.write('\n')
        np.savetxt(func, result, delimiter=",")


if __name__ == '__main__':
    print(tf.__version__)

    #  Network parameter
    NUM_SAMPLE_POINTS = 1024  # Number of input points
    BATCH_SIZE = 16  # Batch size
    EPOCHS = 1000  # Number of epochs
    INITIAL_LR = 1e-3  # Learning rate at the beginning
    NUM_FEATURE = 12  # Number of features:  lokal x, y, z, point normals und global coordinates
    SCALE = 1  # Edge length of the voxels
    SCALE_H = 6  # Vertical edge length of the voxels

    # Experimental parameter
    BOP_ = [False, False, False, False]  # Stack-Optimization
    weighted_loss_a_ = [False, False, False, False]  # wcce_a
    weighted_loss_b_ = [False, False, False, False]  # wcce_b

    used_trained = False  # Load pretrained weights

    Iteration = 10
    name_of_this_experiment = "4-1"  # Name of this experiment

    loaded_weights_ = ["weights/Versuch_" + name_of_this_experiment + "/random/smote/checkpoint/0/",
                       "weights/Versuch_" + name_of_this_experiment + "/random/stack/checkpoint/0/",
                       "weights/Versuch_" + name_of_this_experiment + "/random/wcce_a/checkpoint/0/",
                       "weights/Versuch_" + name_of_this_experiment + "/random/wcce_b/checkpoint/0/"]

    data_train = ["Feature/", "Feature/", "Feature/", "Feature/"]  # original, smote or feature.
    data_test = "Feature/"  # original, smote or feature.

    DATASET_TEST = "versuch_" + name_of_this_experiment + "/"

    DATENSETLISTE = ["versuch_" + name_of_this_experiment + "/",
                     "versuch_" + name_of_this_experiment + "/",
                     "versuch_" + name_of_this_experiment + "/",
                     "versuch_" + name_of_this_experiment + "/"]

    NUM_DATASETS = list(range(0, len(DATENSETLISTE)))  # Number of identical experiments (settings)
    # Load single and independent rooms
    # Training data

    list_41 = ["2015/", "3107/"]

    # List for experiments: "versuch_2-2"

    list_22 = ["0001/", "0003/", "0005/", "0007/", "0010/", "0012/", "4001/", "4003/", "4005/", "4007/", "2015/",
               "3107/"]

    # List for experiments: "versuch_3-#"
    list_32 = ["0001/", "0003/", "0005/", "0007/", "0010/", "0012/", "4001/", "4003/", "4005/", "4007/",
               "4009/", "4011/", "2015/", "3107/"]

    # List for experiments: "versuch_4-2, versuch_4-3 or versuch_4-4"
    list_42 = ["0001/", "0003/", "0005/", "0007/", "0010/", "0012/", "2015/", "3107/"]

    # Combination of list with test data
    dataset_for_experiment = [list_41, list_41, list_41, list_41]

    # Testing data
    # List for experiments: versuch_1, versuch_2-1, versuch_3-1, versuch_4-1, versuch_4-5 or versuch_4-6
    test_list_41 = ["3103/"]

    # List for experiments: versuch_4-2, versuch_4-3 und versuch_4-4
    test_list_42 = ["0004/", "0006/", "3103/"]

    # List for experiments: versuch_3-2 und versuch_3-3
    test_list_32 = ["0004/", "0011/", "4004/", "4010/", "3103/"]

    # List for experiments: versuch_2-2
    test_list_22 = ["0004/", "0011/", "4004/", "3103/"]

    ROOMTESTING_list = test_list_41

    # Evaluation data
    Data_testing = []
    points_test_ = []
    version_of_experiment = name_of_this_experiment + str(0)
    create_dir("Versuch/" + version_of_experiment)
    for rooms_in_testing in ROOMTESTING_list:
        df_test, NUM_CLASSES_LIST_TEST = td.load_folder_pts(data_test + rooms_in_testing + DATASET_TEST)
        NUM_CLASSES = len(NUM_CLASSES_LIST_TEST)
        dataset, points_test = testing_data(df_test, SCALE, SCALE_H, version_of_experiment)
        Data_testing.append(dataset)
        points_test_.append(points_test)

    for nn, DATASET, DATEN, BOP, weighted_loss_a, weighted_loss_b, loaded_weights, trainingsset in zip(NUM_DATASETS,
                                                                                                       DATENSETLISTE,
                                                                                                       data_train,
                                                                                                       BOP_,
                                                                                                       weighted_loss_a_,
                                                                                                       weighted_loss_b_,
                                                                                                       loaded_weights_,
                                                                                                       dataset_for_experiment):

        version_of_experiment_file = name_of_this_experiment + str(nn)
        # Data structure
        create_dir("Versuch/" + version_of_experiment_file)
        images_dir = "Versuch/" + version_of_experiment_file + "/loss_acc/"
        create_dir(images_dir)
        pc_dir = "Versuch/" + version_of_experiment_file + "/pc/"
        create_dir(pc_dir)
        result_filename = 'Versuch/' + version_of_experiment_file + '/'
        create_dir("Versuch/" + version_of_experiment_file + "/checkpoint/")

        with open(result_filename + 'stat.csv', "a") as f:
            f.write('################# Information about' + version_of_experiment_file + ' #################### \n ')

        # Create training data
        df_list = []
        for room_training in trainingsset:
            df1, NUM_CLASSES_LIST = td.load_folder_pts(DATEN + room_training + DATASET)
            NUM_CLASSES = len(NUM_CLASSES_LIST)
            df_list.append(df1)

        train_dataset, val_dataset, total_training_examples, class_weight_a, class_weight_b = training_eval_data(
            df_list, SCALE,
            SCALE_H, version_of_experiment_file)

        # Adaptiv learning rate
        training_step_size = total_training_examples // BATCH_SIZE
        total_training_steps = training_step_size * EPOCHS
        print(f"Total of test steps: {total_training_steps}.")
        lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[training_step_size * 10, training_step_size * 20],
            values=[INITIAL_LR, INITIAL_LR * 0.5, INITIAL_LR * 0.25], )

        if weighted_loss_a == True:
            LOSS_FUNCTION = wcc.weighted_categorical_crossentropy(class_weight_a)
        elif weighted_loss_b == True:
            LOSS_FUNCTION = wcc.weighted_categorical_crossentropy(class_weight_b)
        else:
            LOSS_FUNCTION = keras.losses.CategoricalCrossentropy()

        # Create testing data
        for t in range(0, Iteration):
            VOXEL_SIZE_H = SCALE
            number_of_experiment = str(t)
            create_dir("Versuch/" + version_of_experiment_file + "/checkpoint/" + number_of_experiment + "/")
            segmentation_model, history = run_experiment(EPOCHS,
                                                         "Versuch/" + version_of_experiment_file + "/checkpoint/" +
                                                         number_of_experiment + "/",
                                                         LOSS_FUNCTION, used_trained, loaded_weights)
            plot_result("loss", images_dir, 'loss_' + str(t))
            plot_result("accuracy", images_dir, 'accuracy_' + str(t))

            # Prediction
            try:
                for dataset, room, points_test in zip(Data_testing, ROOMTESTING_list, points_test_):
                    prediction(dataset, t, NUM_CLASSES, result_filename, NUM_SAMPLE_POINTS, SCALE, SCALE_H, room,
                               points_test)

                with open(result_filename + 'results.csv', "a") as f:
                    f.write('################################## \n')
            except:
                print('No model found! No Prediction')
                continue
