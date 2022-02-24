# Keras-to-ONNX-Conversion
Conversion Pytorch/Keras/Tensorflow into ONNX model format using ResNet50
## About the project

First the following model is created using Python and then the Python inference is being used to convert the following into the required 
Conversion of the Pytorch/Tensorflow/keras model into ONNX format and then implement the face detection and face recognition in Unity. Keras models can be converted into unity using the keras2onnx package. Here the MCNN python inference is being used for the conversion. For the following conversion into the required package.
Installing the Barracuda package in order to implement the following ONNX model in Unity and then use it as an inference. 

Using OpenCV dependencies for the face detection and face recognition which is used to represent the features of the face and bounding boxes to define the boundaries.

Conversion of model to the ResNet model. ResNet models perform image classification - they take images as input and classify the major object in the image into a set of pre-defined classes. They are trained on ImageNet dataset which contains images from 1000 classes. ResNet models provide very high accuracies with affordable model sizes. They are ideal for cases when high accuracy of classification is required.

## Barracuda 
The Barracuda package is a lightweight cross-platform neural network inference library for Unity.

Barracuda can run neural networks on both the GPU and CPU. For details, see Supported platforms.

Currently Barracuda is production-ready for use with machine learning (ML) agents and number of other network architectures. When you use Barracuda in other scenarios, it is in the preview development stage.

The Barracuda neural network import pipeline is built on the ONNX (Open Neural Network Exchange) format, which lets you bring in neural network models from a variety of external frameworks, including Pytorch, TensorFlow, and Keras.

Installing the Barracuda packages from the package manager in Unity.

## ONNX
ONNX(Open Neural Network Exchange) is an open format built to represent machine learning models. ONNX defines a common set of operators - the building blocks of machine learning and deep learning models - and a common file format to enable AI developers to use models with a variety of frameworks, tools, runtimes, and compilers. This is a standard format to import machine learning models into Unity.

tflite conversion for Python Level API and then converting into tensorflow lite model then abbreviating the following into ONNX using the tflite converter.

## Getting Started

### Pre-requisites and Installation
1) Run the train.py
2) Run the detect.py(test)
3)To convert h5 to ONNX, run keras2onnx-example-master/convert_keras_to_onnx.py

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all the necessary packages.

```bash
!pip install tf2onnx
!pip install facenet-torch
!pip install onnx
!pip install torchvision
!pip install h5py
!pip install onnx_tf
```

## Usage
The traditional face recognition model implemented via using Python API is now converted to ONNX to be implemented in Unity.

```python
import os 
import cv2
import mtcnn
import pickle 
import numpy as np 
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import load_model


import onnx
from onnx_tf.backend import prepare

```

### Unity Interdependencies
```
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.ObjdetectModule;
using OpenCVForUnity.UnityUtils;
using UnityEngine;
using UnityEngine.UI;
```
## 
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Acknowledgement

1) ONNX 
2) Barracuda
3) Face Detection and Face Recognition
