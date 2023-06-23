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
