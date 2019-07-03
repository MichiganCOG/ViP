import torch
from .abstract_datasets import RecognitionDataset 
from PIL import Image
import cv2
import os
import numpy as np
import datasets.preprocessing_transforms as pt
from torchvision import transforms

class HMDB51(RecognitionDataset):
    def __init__(self, *args, **kwargs):
        super(HMDB51, self).__init__(*args, **kwargs)

        self.load_type    = kwargs['load_type']
        self.resize_shape = kwargs['resize_shape']
        self.crop_shape   = kwargs['crop_shape']
        self.final_shape  = kwargs['final_shape']
        
        if self.load_type=='train':
            self.transforms = PreprocessTrainC3D(**kwargs)

        else:
            self.transforms = PreprocessEval(**kwargs)

    def __getitem__(self, idx):
        vid_info  = self.samples[idx]
        base_path = vid_info['base_path']

        input_data = []
        vid_data   = np.zeros((self.clip_length, self.final_shape[0], self.final_shape[1], 3))-1
        labels     = np.zeros((self.clip_length))-1
        input_data = []
    
        for frame_ind in range(len(vid_info['frames'])):
            frame_path   = os.path.join(base_path, vid_info['frames'][frame_ind]['img_path'])

            for frame_labels in vid_info['frames'][frame_ind]['actions']:
                labels[frame_ind] = frame_labels['action_class']

            # Load frame image data and preprocess image accordingly
            input_data.append(cv2.imread(frame_path)[...,::-1]/255.)
            #input_data.append(Image.open(frame_path))


        # Preprocess data
        vid_data   = self.transforms(input_data)
        labels     = torch.from_numpy(labels).float()

        ## Permute the PIL dimensions (Frame, Height, Width, Chan) to pytorch (Chan, frame, height, width) 
        vid_data = vid_data.permute(3, 0, 1, 2)

        ret_dict           = dict() 
        ret_dict['data']   = vid_data 
        ret_dict['labels'] = labels

        return ret_dict


class PreprocessTrainC3D(object):
    """
    Container for all transforms used to preprocess clips for training in this dataset.
    """
    def __init__(self, **kwargs):
        crop_type = kwargs['crop_type']
        self.transforms = []

        self.clip_mean = np.zeros((16,128,171,3), dtype='float')

        self.transforms.append(pt.ResizeClip(**kwargs))
        self.transforms.append(pt.ToTensorClip(**kwargs))
        self.transforms.append(pt.SubtractMeanClip(clip_mean=self.clip_mean, **kwargs))
        

        if crop_type == 'Random':
            self.transforms.append(pt.RandomCropClip(**kwargs))
        else:
            self.transforms.append(pt.CenterCropClip(**kwargs))
        
        self.transforms.append(pt.RandomFlipClip(direction='h', p=1.0, **kwargs))
        self.transforms.append(pt.RandomRotateClip(**kwargs))
        self.transforms.append(pt.ToTensorClip(**kwargs))


    def __call__(self, input_data):
        """
        Preprocess the clip accordingly
        Args:
            input_data: List of PIL images containing clip frames 

        Return:
            input_data: Pytorch tensor containing the processed clip data 
        """
        for transform in self.transforms:
            input_data = transform(input_data)

        return input_data

class PreprocessTrain(object):
    """
    Container for all transforms used to preprocess clips for training in this dataset.
    """
    def __init__(self, **kwargs):
        crop_shape = kwargs['crop_shape']
        crop_type = kwargs['crop_type']
        resize_shape = kwargs['resize_shape']
        self.transforms = []

        if crop_type == 'Random':
            self.transforms.append(pt.RandomCropClip(*crop_shape))
        else:
            self.transforms.append(pt.CenterCropClip(*crop_shape))

        self.transforms.append(pt.ResizeClip(*resize_shape))
        self.transforms.append(pt.RandomFlipClip(direction='h', p=1.0))
        self.transforms.append(pt.RandomRotateClip())
        self.transforms.append(pt.ToTensorClip())


    def __call__(self, input_data):
        """
        Preprocess the clip accordingly
        Args:
            input_data: List of PIL images containing clip frames 

        Return:
            input_data: Pytorch tensor containing the processed clip data 
        """
        for transform in self.transforms:
            input_data = transform(input_data)

        return input_data


class PreprocessEval(object):
    """
    Container for all transforms used to preprocess clips for evaluation in this dataset.
    """
    def __init__(self):
        self.resize   = pt.ResizeClip(128,171)
        self.crop     = pt.CenterCropClip(112, 112)
        self.flip     = pt.RandomFlipClip(direction='h')
        self.toTensor = pt.ToTensorClip()
        #self.mean     = np.load('models/weights/sport1m_train16_128_mean.npy')[0]
        #self.mean     = torch.Tensor(np.transpose(self.mean, (1,2,3,0)))
    def __init__(self, **kwargs):
        crop_shape = kwargs['crop_shape']
        crop_type = kwargs['crop_type']
        resize_shape = kwargs['resize_shape']
        self.transforms = []

        if crop_type == 'Random':
            self.transforms.append(pt.RandomCropClip(*crop_shape))
        else:
            self.transforms.append(pt.CenterCropClip(*crop_shape))

        self.transforms.append(pt.ResizeClip(*resize_shape))
        self.transforms.append(pt.ToTensorClip())


    def __call__(self, input_data):
        """
        Preprocess the clip and the bbox data accordingly
        Args:
            input_data: List of PIL images containing clip frames 
            bbox_data:  Numpy array containing bbox coordinates per object per frame 

        Return:
            input_data: Pytorch tensor containing the processed clip data 
            bbox_data:  Numpy tensor containing the augmented bbox coordinates
        """
        input_data = self.resize(input_data)
        input_data = self.crop(input_data)
        input_data = self.flip(input_data)
        input_data = self.toTensor(input_data)

        return input_data



#dataset = HMDB51(json_path='/z/dat/HMDB51', dataset_type='train', clip_length=100, num_clips=0)
#dat = dataset.__getitem__(0)
#import pdb; pdb.set_trace()
