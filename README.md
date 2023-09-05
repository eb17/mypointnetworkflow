# mypointnetworkflow

### Introduction
This source code was created as part of my dissertation at HafenCity University Hamburg. The collection of scripts is 
used to examine dataset-specific properties and hyperparameters of artificial neural networks. The data preprocessing 
is optimized for the deep learning method PointNet. 

PointNets from <a href="https://github.com/charlesq34/pointnet" target="_blank">Charles R. Qi</a> original source code,
was one of the fist point-based DeepLearning antilogarithm. It is adjusted many times. I use the keras-base sample code 
from <a href="https://dgriffiths3.github.io" target="_blank">David Griffiths</a> as base for my experiments.

The weighted loss function has been developed on the basis of the post of <a href="https://stackoverflow.com/questions/59520807/multi-class-weighted-loss-for-semantic-image-segmentation-in-keras-tensorflow" target="_blank">Mendi Barel</a> and is a script in this repository.  

All lines taken from other source code are marked in the respective scripts. Links to posts and source code, which were idea-generating or from which the source code was taken, are marked in my source code. 
The source code has been created and documented to the best of my knowledge. A guarantee and liability for the source code stallige damage is not taken over by the author. The user acts at his own risk. The code was created without the use of artificial intelligence. 

The main script is 

    training_run.py. 

In this script the hyperparameters and investigation parameters are defined.

For creating test and training data use the scrips in the folder:
    
    Data preparation

The network architecture can be found in script:
    
    pointnet_keras.py

The easy to use wcce function is defined in 

    weighted_categorical_crossentropy.py

### Installation

Install <a href="https://www.tensorflow.org/get_started/os_setup" target="_blank">TensorFlow</a>.
You also need to install matplotlib, pandas, numpy, open3d and sklearn, pyntcloud. 
The code has been tested with Python 3.8.3, TensorFlow 2.3.0, CUDA 11.0 and cuDNN V10.1.243 on Windows10. 

Use your favourite terminal and install packages for Python:
```bash
 pip3 install *package*
```

### License
The code is released under MIT License (see LICENSE file for details), unless other restrictions prohibit it.

### Citation
If you find our work useful in your research, please consider citing:

	@PhdThesis{Barnefske2023,
	  title={Automated segmentation and classification with artificial neural networks of objects in 3D point clouds},
	  author={Barnefske,Eike},
      date        = {2023},
      institution = {HafenCity University Hamburg}, 
      type        = {phdthesis},
    }
