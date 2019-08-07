# Video Platform for Recognition and Detection in Pytorch

A platform for quick and easy development of deep learning networks for recognition and detection in videos. Includes popular models like C3D and SSD.

## Implemented Models and their performance

### Recognition
|  Model Architecture  |      Dataset       |    ViP Accuracy (%)   |  
|:--------------------:|:------------------:|:---------------------:|
|        C3D           |  HMDB51 (Split 1)  |    50.14 ± 0.777      |
|        C3D           |  UCF101 (Split 1)  |    80.40 ± 0.399      |


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
virtualenv -p python3.6 --no-site-packages vip
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


Ex: From the root directory of ViP, evaluate the action recognition network C3D on HMDB51
```
python eval.py --cfg_file models/c3d/config_test.yaml
```

### Training

Run `train.py` with the argument `--cfg_file` pointing to the desired model config yaml file.

Ex: From the root directory of ViP, train the action recognition network C3D on HMDB51
```
python train.py --cfg_file models/c3d/config_train.yaml
```
## Development

New models and datasets can be added without needing to rewrite any training, evaluation, or data loading code.

### Add a Model

To add a new model:
1. Create a new folder `ViP/models/custom_model_name` 
2. Create a model class in `ViP/models/custom_model_name/custom_model_name.py`
	* Complete `__init__`, `forward`, and (optional) `__load_pretrained_weights` functions
3. Add PreprocessTrain and PreprocessEval classes within `custom_model_name.py`
4. Create `config_train.yaml` and `config_test.yaml` files for the new model

Examples of previously implemented models can be found [here](https://github.com/MichiganCOG/ViP/tree/master/models).

### Add a Dataset

To add a new dataset:
1. Convert annotation data to our JSON format
	* The JSON skeleton templates can be found [here](https://github.com/MichiganCOG/ViP/tree/master/datasets/templates)
	* Existing scripts for datasets can be found [here](https://github.com/MichiganCOG/ViP/tree/master/datasets/scripts)
2. Create a dataset class in `ViP/datasets/custom_dataset_name.py`.
	* Inherit `DetectionDataset` or `RecognitionDataset` from `ViP/abstract_dataset.py`
	* Complete `__init__` and `__getitem__` functions
	* Example skeleton dataset can be found [here](https://github.com/MichiganCOG/ViP/blob/master/datasets/templates/dataset_template.py)

## Version History

### Version 1.0
- Training and Evaluation files are completed
- Data loading pipeline
- Config and argument reader
- Checkpoint saving and loading
- Implemented datasets: HMDB51, ImageNetVID, MSCOCO, VOC2007
- Implemented recognition models: C3D
- Implemented detection models: SSD
- Implemented metrics: Recognition accuracy, IOU, mAP, Precision, Recall
- Implemented losses: MSE, Cross entropy

### Version 0.1
- Basic training file for recognition is ready, next step is to work on data loader since that is the immediate function being loaded
- Began inclusion of dataset loading classes into datasets folder

