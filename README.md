# A Performance Evaluation of Several Deep Neural Networks for Reverberant Speech Separation
Qingju Liu, Wenwu Wang, Philip JB Jackson, Saeid Safavi 

[Paper](http://epubs.surrey.ac.uk/849431/) in Asilomar Conference on Signals, Systems, and Computers, Pacific Grove, CA, US, October, 2018

In this paper, we compare different deep neural networks (DNN) in extracting speech signals from competing speakers in room environments, including the conventional fully-connected multilayer perception (MLP) network, convolutional neural network (CNN), recurrent neural network (RNN), and the recently proposed capsule network (CapsNet). Each DNN takes input of both spectral features and converted spatial features that are robust to position mismatch, and outputs the separation mask for target source estimation. In addition, a psychacoustically-motivated objective function is integrated in each DNN, which explores perceptual importance of each TF unit in the training process. Objective evaluations are performed on the separated sounds using the converged models, in terms of PESQ, SDR as well as STOI. Overall, all the implemented DNNs have greatly improved the quality and speech intelligibility of the embedded target source as compared to the original recordings. In particular, bidirectional RNN, either along the temporal direction or along the frequency bins, outperforms the other DNN structures with consistent improvement. 

******************************************************************************************
This code is implemented in Python, using [Keras](https://keras.io/) with [Tensorflow](https://www.tensorflow.org/) backend where "image_data_format": "channels_last" was set.


## Data preparation

To train the source separation models, you need to prepare the training data and two training parameters.

The feature extraction and parameter estimation process for each simulated binaural mixture can be hinted from DataGenerator.py, which can be easily implemented in Matlab


## Train

Main code in main.py

## Test

Apply source separation after the DNN model is trained.

Example code in TestModel.py
