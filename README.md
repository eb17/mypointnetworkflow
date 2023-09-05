# mypointnetworkflow
*In this repository the developed code will be published after the PhD thesis is published.
You find a beta version here*

Created by Eike Barnefske from HafenCity University Hamburg.

### Introduction
This source code was created as part of my dissertation at HafenCity University Hamburg. The collection of scripts is 
used to examine dataset-specific properties and hyperparameters of artificial neural networks. The data preprocessing 
is optimized for the deep learning method PointNet. 

PointNet in the variant, of the example of xxx and xxx (<a href="http://charlesrqi.com" target="_blank">Keras</a>) is,
the basic network. This was adapted in a few places for our purposes. The adapted version of PointNet is part of this
repository as a script. 

The weighted loss function has been developed on the basis of the post of <a href="https://stackoverflow.com/questions/59520807/multi-class-weighted-loss-for-semantic-image-segmentation-in-keras-tensorflow" target="_blank">Mendi Barel</a> and is a script in this repository.  

The main script is xxx. In this script the hyperparameters and investigation parameters are defined. Further, scripts for data enhancement can be found in the folder xxx. 
### Citation
If you find our work useful in your research, please consider citing:

	@article{Barnefske2023,
	  title={Automated segmentation and classification with artificial neural networks of objects in 3D point clouds},
	  author={Barnefske,Eike},
	  journal={xx},
	  year={2023}
	}
   
### Installation

Install <a href="https://www.tensorflow.org/get_started/os_setup" target="_blank">TensorFlow</a>.
You also need to install matplotlib, pandas, numpy, open3d and sklearn, pyntcloud. 
The code has been tested with Python 3.8.3, TensorFlow 2.3.0, CUDA 11.0 and cuDNN V10.1.243 on Windows10. 

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

### License
Our code is released under MIT License (see LICENSE file for details).
