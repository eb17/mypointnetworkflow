# mypointnetworkflow
*In this repository the developed code will be published after the PhD thesis is published.
You find a beta version here*

Created by Eike Barnefske from HafenCity University Hamburg.

### Introduction
This collection of scripts is used to examine dataset-specific properties and hyperparameters of artificial neural networks. The data preprocessing is optimized for the deep learning method PointNet. PointNet in the variant, of the example of xxx and xxx (<a href="http://charlesrqi.com" target="_blank">Keras</a>) is, the basic network. This was adapted in a few places for our purposes. The adapted version of PointNet is part of this repository as a script. 

The weighted loss function has been developed on the basis of the post of B. and is a script in this repository.  

The main script is xxx. In this script the hyperparameters and investigation parameters are defined. Further, scripts for data enhancement can be found in the folder xxx. 
### Citation
If you find our work useful in your research, please consider citing:

	@article{Barnefske2023,
	  title={xxxSegmentation},
	  author={Barnefske,Eike},
	  journal={xx},
	  year={2023}
	}
   
### Installation

Install <a href="https://www.tensorflow.org/get_started/os_setup" target="_blank">TensorFlow</a>. You also need to install matplotlib, pandas, numpy, open3d and sklearn, pyntcloud. The code has been tested with Python 3.8.3, TensorFlow 2.3.0, CUDA 11.0 and cuDNN V10.1.243 on Windows10. 

Use your favourite terminal and install packages for Python:
```bash
 pip3 install *package*
```

### Usage
To train a model to classify point clouds sampled from 3D shapes:

    python train.py

Log files and network parameters will be saved to `log` folder in default. Point clouds of <a href="http://modelnet.cs.princeton.edu/" target="_blank">ModelNet40</a> models in HDF5 files will be automatically downloaded (416MB) to the data folder. Each point cloud contains 2048 points uniformly sampled from a shape surface. Each cloud is zero-mean and normalized into an unit sphere. There are also text files in `data/modelnet40_ply_hdf5_2048` specifying the ids of shapes in h5 files.

To see HELP for the training script:

    python train.py -h

We can use TensorBoard to view the network architecture and monitor the training progress.

    tensorboard --logdir log

After the above training, we can evaluate the model and output some visualizations of the error cases.

    python evaluate.py --visu

Point clouds that are wrongly classified will be saved to `dump` folder in default. We visualize the point cloud by rendering it into three-view images.

If you'd like to prepare your own data, you can refer to some helper functions in `utils/data_prep_util.py` for saving and loading HDF5 files.

### Part Segmentation
To train a model for object part segmentation, firstly download the data:

    cd part_seg
    sh download_data.sh

The downloading script will download <a href="http://web.stanford.edu/~ericyi/project_page/part_annotation/index.html" target="_blank">ShapeNetPart</a> dataset (around 1.08GB) and our prepared HDF5 files (around 346MB).

Then you can run `train.py` and `test.py` in the `part_seg` folder for training and testing (computing mIoU for evaluation).

### License
Our code is released under MIT License (see LICENSE file for details).

### Selected Projects that Use PointNet

* <a href="http://stanford.edu/~rqi/pointnet2/" target="_blank">PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space</a> by Qi et al. (NIPS 2017) A hierarchical feature learning framework on point clouds. The PointNet++ architecture applies PointNet recursively on a nested partitioning of the input point set. It also proposes novel layers for point clouds with non-uniform densities.
* <a href="http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w13/Engelmann_Exploring_Spatial_Context_ICCV_2017_paper.pdf" target="_blank">Exploring Spatial Context for 3D Semantic Segmentation of Point Clouds</a> by Engelmann et al. (ICCV 2017 workshop). This work extends PointNet for large-scale scene segmentation.
* <a href="https://arxiv.org/abs/1710.04954" target="_blank">PCPNET: Learning Local Shape Properties from Raw Point Clouds</a> by Guerrero et al. (arXiv). The work adapts PointNet for local geometric properties (e.g. normal and curvature) estimation in noisy point clouds.
* <a href="https://arxiv.org/abs/1711.06396" target="_blank">VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection</a> by Zhou et al. from Apple (arXiv) This work studies 3D object detection using LiDAR point clouds. It splits space into voxels, use PointNet to learn local voxel features and then use 3D CNN for region proposal, object classification and 3D bounding box estimation.
* <a href="https://arxiv.org/abs/1711.08488" target="_blank">Frustum PointNets for 3D Object Detection from RGB-D Data</a> by Qi et al. (arXiv) A novel framework for 3D object detection with RGB-D data. The method proposed has achieved first place on KITTI 3D object detection benchmark on all categories (last checked on 11/30/2017).
