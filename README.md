# Video Platform for Recognition and Detection in Pytorch

A platform for quick and easy development of deep learning networks for recognition and detection in videos. Includes popular models like C3D and SSD.

Check out our [wiki!](https://github.com/MichiganCOG/ViP/wiki)

## Implemented Models and their performance

### Recognition
|  Model Architecture  |      Dataset       |    ViP Accuracy (%)   |  
|:--------------------:|:------------------:|:---------------------:|
|        I3D           |  HMDB51 (Split 1)  |    72.75              |
|        C3D           |  HMDB51 (Split 1)  |    50.14 ± 0.777      |
|        C3D           |  UCF101 (Split 1)  |    80.40 ± 0.399      |

### Object Detection
|  Model Architecture  |      Dataset       |    ViP Accuracy (%)   | 
|:--------------------:|:------------------:|:---------------------:|
|        SSD300        |  VOC2007  |    76.58      |

### Video Object Grounding
|  Model Architecture  |      Dataset       |    ViP Accuracy (%)   | 
|:--------------------:|:------------------:|:---------------------:|
|        DVSA (+fw, obj)        |  YC2-BB (Validation)  |    30.09      |

**fw**: framewise weighting, **obj**: object interaction
## Table of Contents

* [Datasets](#configured-datasets)
* [Models](#models)
* [Requirements](#requirements)
* [Installation](#installation)
* [Quick Start](#quick-start)
  * [Testing](#testing)
  * [Training](#training)
* [Development](#development)
  * [Add a Model](#add-a-model)
  * [Add a Dataset](#add-a-dataset)
* [FAQ](#faq)

## Configured Datasets
|   Dataset      |        Task(s)           |
|:--------------:|:------------------------:|
|[HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)      | Activity Recognition   |
|[UCF101](https://www.crcv.ucf.edu/data/UCF101.php)                                                    | Activity Recognition   |
|[ImageNetVID](http://bvisionweb1.cs.unc.edu/ilsvrc2015/download-videos-3j16.php)                      | Video Object Detection |
|[MSCOCO 2014](http://cocodataset.org/#download)                                                       | Object Detection, Keypoints|
|[VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)                                            | Object Detection, Classification|
|[YC2-BB](http://youcook2.eecs.umich.edu/download)| Video Object Grounding|
|[DHF1K](https://github.com/wenguanwang/DHF1K)							       | Video Saliency Prediction|

## Models
|                     Model                        |        Task(s)       |
|:------------------------------------------------:|:--------------------:|
|[C3D](https://github.com/jfzhang95/pytorch-video-recognition/blob/master/network/C3D_model.py) | Activity Recognition |
|[I3D](https://github.com/piergiaj/pytorch-i3d) | Activity Recognition |
|[SSD300](https://github.com/amdegroot/ssd.pytorch)                                             | Object Detection     |
|[DVSA (+fw, obj)](https://github.com/MichiganCOG/Video-Grounding-from-Text)| Video Object Grounding|

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
Run `train.py` and `eval.py` to train or test any implemented model. The parameters of every experiment is specified in its [config.yaml](https://github.com/MichiganCOG/ViP/blob/master/config_default_example.yaml) file. 

Use the `--cfg_file` command line argument to point to a different config yaml file.
Additionally, all config parameters can be overriden with a command line argument.

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

Additional examples can be found on our [wiki.](https://github.com/MichiganCOG/ViP/wiki)

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

Additional information can be found on our [wiki.](https://github.com/MichiganCOG/ViP/wiki)

### Add a Dataset

To add a new dataset:
1. Convert annotation data to our JSON format
	* The JSON skeleton templates can be found [here](https://github.com/MichiganCOG/ViP/tree/master/datasets/templates)
	* Existing scripts for datasets can be found [here](https://github.com/MichiganCOG/ViP/tree/master/datasets/scripts)
2. Create a dataset class in `ViP/datasets/custom_dataset_name.py`.
	* Inherit `DetectionDataset` or `RecognitionDataset` from `ViP/abstract_dataset.py`
	* Complete `__init__` and `__getitem__` functions
	* Example skeleton dataset can be found [here](https://github.com/MichiganCOG/ViP/blob/master/datasets/templates/dataset_template.py)

Additional information can be found on our [wiki.](https://github.com/MichiganCOG/ViP/wiki)

### FAQ

A detailed FAQ can be found on our [wiki](https://github.com/MichiganCOG/ViP/wiki/FAQ).
