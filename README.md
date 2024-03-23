# MTLDOG - Domain Generalization for Multi-task Learning

## Overview

MTLDOG stands for Multi-Task Learning with Domain Generalization. It is designed to facilitate research and experimentation in the field of multi-task learning, particularly focusing on domain generalization scenarios.

### Features

- **CLI Interaction**: Easily interact with MTLDOG via the command line interface.
- **Flexible Configuration**: Customize various aspects of training, including datasets, tasks, losses, methods, models, and training parameters.
- **Domain Generalization**: Incorporate domain generalization techniques into multi-task learning training pipelines.
- **Logging and Monitoring**: Utilize logging by using WandB for online and offline saving of training progress.

### Usage

To use MTLDOG, simply run the `main.py` script with appropriate command-line arguments. Here's an example of how to run MTLDOG:

```
python main.py -h
```

### Requirements
```
alive-progress==3.1.5
numpy==1.26.4
opencv-python==4.9.0.80
pillow==10.2.0
PyYAML==6.0.1
tensorboard==2.16.2
wandb==0.16.4
albumentations==1.4.1
pandas==2.2.1
torch  # Add PyTorch separately as it may have version-specific installation requirements
```

### Reference
In this part we list all previous projects that did help us in figuring out coding issues in MTLDOG.

## Table of Contents
- [Overview](#overview)
  - [Features](#features)
  - [Usage](#usage)
  - [Requirements](#requirements)
  - [Reference](#reference)
- [Dataset](#dataset)
- [Training](#training)
  - [Common Parameters](#common-parameters)
  - [Algorithm Hyper-parameters](#algorithm-hyper-parameters)
  - [Default Scripts](#default-scripts)
  - [Hyper-parameter Sweep Search](#hyper-parameter-sweep-search)
- [Logging](#logging)
- [Citation](#citation)

## Dataset
MTLDOG supports various datasets for multi-task learning experiments. Below are some of the datasets currently available:

### RotateMnist Dataset

The RotateMnist dataset consists of rotated MNIST images, where each image is rotated by a certain degree. It is commonly used for evaluating the performance of multi-task learning models on rotation-invariant tasks.

### CityScapes Dataset

The CityScapes dataset is a large-scale dataset for semantic urban scene understanding. It contains high-quality pixel-level annotations for urban street scenes, making it suitable for various tasks in computer vision and scene understanding. Some of the tasks supported by CityScapes include:

- [x] **Semantic Segmentation**: Predicting the semantic labels of pixels in urban street scenes.
- [x] **Depth Estimation**: Estimating the depth or distance of objects in the scene from the camera viewpoint.
- [ ] **Instance Segmentation**: Identifying and segmenting individual objects within urban scenes.
- [ ] **Human Detection**: Detecting and localizing human instances in urban scenes.
- [ ] **3D Object Detection**: Predicting the 3D bounding boxes of objects present in the scene, providing information about their position and size in 3D space.


## Training

### Common Parameters

### Algorithm Hyper-parameters

### Default Scripts

### Hyper-parameter Sweep Search

## Logging

# Citation
If you find this project useful for your research, consider cite it to your research paper.
```
```