# <span style="color:purple">M</span><span style="color:purple">T</span><span style="color:purple">L</span><span style="color:purple">D</span><span style="color:purple">O</span><span style="color:purple">G</span> - <span style="color:purple">D</span>omain <span style="color:purple">G</span>eneralization for <span style="color:purple">M</span>ulti-<span style="color:purple">T</span>ask <span style="color:purple">L</span>earning


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
<details>
  <summary>Detail Information</summary>

The RotateMnist dataset is commonly used for evaluating multi-task learning models on digit classification and image reconstruction tasks across various domains. It consists of rotated MNIST images, where each image is rotated by a certain degree. The dataset provides a challenging setting for multi-task learning, with tasks including:

#### Tasks

- [x] **Digit Classification**: Predicting the digit label of each image.
- [x] **Image Reconstruction**: Reconstructing the original image from its rotated versions.

#### Domains

The dataset provides three versions of increasing difficulty, each with different numbers of domains:

- [x] **Easy (5 Domains)**: Contains images rotated across 5 domains.
- [x] **Medium (6 Domains)**: Contains images rotated across 6 domains.
- [x] **Hard (10 Domains)**: Contains images rotated across 10 domains.

#### Download
This dataset will be automatically downloaded when conducting the experiments. 
</details>

### CityScapes Dataset
<details>
  <summary>Detail Information</summary>
The CityScapes dataset is a large-scale dataset for semantic urban scene understanding. It contains high-quality pixel-level annotations for urban street scenes, making it suitable for various tasks in computer vision and scene understanding. 

#### Tasks

Some of the tasks supported by CityScapes include:
- [x] **Semantic Segmentation**: Predicting the semantic labels of pixels in urban street scenes.
- [x] **Depth Estimation**: Estimating the depth or distance of objects in the scene from the camera viewpoint.
- [ ] **Instance Segmentation**: Identifying and segmenting individual objects within urban scenes.
- [ ] **Human Detection**: Detecting and localizing human instances in urban scenes.
- [ ] **3D Object Detection**: Predicting the 3D bounding boxes of objects present in the scene, providing information about their position and size in 3D space.

#### Domains
The dataset includes annotations for various environmental conditions:

- [x] **Clear**: Scenes captured under clear weather conditions.
- [x] **Foggy**: Scenes captured under foggy weather conditions.
- [x] **Rainy**: Scenes captured under rainy weather conditions.

#### Download

</details>

## Training
### Common Parameters
<details>
  <summary>Detail Information</summary>

| Parameter     | Description                                                                                       |
|---------------|---------------------------------------------------------------------------------------------------|
| --ds          | Dataset in use.                                                                                   |
| --dt          | Root data directory.                                                                              |
| --bs          | Batch size.                                                                                       |
| --wk          | Number of dataset workers.                                                                        |
| --pm          | Toggle to use pin memory.                                                                         |
| --trdms       | List of domains used in training.                                                                 |
| --tkss        | List of tasks used in training.                                                                   |
| --losses      | Losses of tasks used in training.                                                                 |
| --m           | Method used in training.                                                                          |
| --hp          | JSON file path for hyper-parameters of method.                                                    |
| --model       | Model type (e.g., ae, hps (hard parameter sharing)).                                             |
| --at          | Architecture type (e.g., ae, unet).                                                               |
| --bb          | Backbone type (e.g., ae, base, resnet18).                                                         |
| --seed        | Seed for random number generation.                                                                |
| --tm          | Training mode (e.g., sup (supervised)).                                                           |
| --dvids       | List of device IDs used in training.                                                              |
| --epoch       | Number of epochs used in training.                                                                |
| --lr          | Learning rate.                                                                                    |
| --port        | Multi-GPU Training Port.                                                                          |
| --wandb       | Toggle to use wandb for online saving.                                                            |
| --log         | Toggle to use tensorboard for offline saving.                                                     |
| --wandb_prj   | Wandb project name.                                                                               |
| --wandb_entity| Wandb entity name.                                                                                |
| --gamma       | Gamma for focal loss.                                                                             |

</details>

### Algorithm Hyper-parameters

### Default Scripts

### Hyper-parameter Sweep Search

## Logging

# Citation
If you find this project useful for your research, consider cite it to your research paper.
```
```