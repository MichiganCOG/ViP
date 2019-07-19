# Video Platorm for Recognition and Detection in Pytorch

A platform for quick and easy development of deep learning networks for recognition and detection in videos. Includes popular models like C3D, ResNet, and SSD.

## Table of Contents

* [Requirements](#requirements)
* [Installation](#installation)
* [Quick Start](#quick-start)
  * [Testing](#testing)
  * [Training](#training)
* [Development](#development)
  * [Add a Model](#add-a-model)
  * [Add a Dataset](#add-a-dataset)
* [Version History](#version-history)

## Requirements

* Python 3.6
* Cuda 9.0
* (Suggested) Virtualenv

## Installation

```
# Set up Python3 virtual environment
virtualenv -p python3.6 vip
source vip/bin/activate

# Clone ViP repository
git clone https://github.com/MichiganCOG/ViP
cd ViP

# Install requirements and model weights
./install.sh
```

## Quick Start
Eval.py and train.py are the two programs that will test and train any implemented model.
Each model has specific parameters specified within the [config.yaml](https://github.com/MichiganCOG/ViP/blob/master/models/ssd/config.yaml) files in its repsective folder.
All parameters can also be modified using command line arguments.

### Testing

Run `eval.py` with the argument `--cfg_file` pointing to the desired model config yaml file.


Ex: From the root directory of ViP, evaluate the object detection network SSD on VOC2007
```
python eval.py --cfg_file models/c3d/config_test.py
```

### Training

Run `train.py` with the argument `--cfg_file` pointing to the desired model config yaml file.

Ex: From the root directory of ViP, train the action recognition network C3D on HMDB51
```
python train.py --cfg_file models/c3d/config_train.py
```
## Development

New models and datasets can be added without needing to rewrite any training, evaluation, or data loading code.

### Add a Model

To add a new model, the only requirement for the user is to add a new folder to `ViP/models/`. 
Within this folder you will specify the desired `model.py` as well as the `config_train.yaml` and `config_test.yaml` for the new model.
Examples of previously implemented models can be found [here](https://github.com/MichiganCOG/ViP/tree/master/models).

### Add a Dataset

To add a new dataset, the user must only create a single file `ViP/datasets/custom_dataset_name.py`.
Within this file, the user must specify the dataset dataloader class containing `__init__` and `__getitem__` methods. 
This class must inherit from `DetectionDataset` or `RecognitionDataset` in `ViP/abstract_datasets.py` .
Within `__getitem__`.

## Version History

### Version 1.0
- Training and Evaluation files are completed
- Data loading pipeline
- Config and argument reader
- Checkpoint saving and loading
- Implemented datasets: HMDB51, ImageNetVID, MSCOCO, VOC2007
- Implemented recognition models: C3D, ResNet3D
- Implemented detection models: SSD
- Implemented metrics: Recognition accuracy, IOU, mAP, Precision, Recall
- Implemented losses: MSE, Cross entropy

### Version 0.1
- Basic training file for recognition is ready, next step is to work on data loader since that is the immediate function being loaded
- Began inclusion of dataset loading classes into datasets folder

