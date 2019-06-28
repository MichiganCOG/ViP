import torch
from .abstract_datasets import DetectionDataset 
from PIL import Image
import os
import numpy as np
import datasets.preprocessing_transforms as pt

class VOC2007(DetectionDataset):

    def __init__(self, *args, **kwargs):
        super(VOC2007, self).__init__(*args, **kwargs)

        self.load_type = kwargs['load_type']

        # Maximum number of annotated object present in a single frame in entire dataset
        # Dictates the return size of annotations in __getitem__
        self.max_objects = 20
        #Map class name to a class id
        self.class_to_id = {'person':0,
                            'bird':1,
                            'cat':2,
                            'cow':3,
                            'dog':4,
                            'horse':5,
                            'sheep':6,
                            'aeroplane':7,
                            'bicycle':8,
                            'boat':9,
                            'bus':10,
                            'car':11,
                            'motorbike':12,
                            'train':13,
                            'bottle':14,
                            'chair':15,
                            'diningtable':16,
                            'pottedplant':17,
                            'sofa':18,
                            'tvmonitor':19
                            }
        #TODO: maybe add a reverse mapping

        if self.load_type=='train':
            self.transforms = PreprocessTrain(**kwargs)

        else:
            self.transforms = PreprocessEval(**kwargs)


    def __getitem__(self, idx):
        vid_info = self.samples[idx]
        
        base_path = vid_info['base_path']
        vid_size  = vid_info['frame_size']

        input_data = []
        vid_data   = np.zeros((self.clip_length, self.final_shape[0], self.final_shape[1], 3))-1
        bbox_data  = np.zeros((self.clip_length, self.max_objects, 4))-1
        labels     = np.zeros((self.clip_length, self.max_objects))-1

        for frame_ind in range(len(vid_info['frames'])):
            frame      = vid_info['frames'][frame_ind]
            frame_path = frame['img_path']
            
            # Extract bbox and label data from video info
            for obj in frame['objs']:
                trackid   = obj['trackid']
                label     = self.class_to_id[obj['c']]
                obj_bbox  = obj['bbox'] # [xmin, ymin, xmax, ymax]
                
                bbox_data[frame_ind, trackid, :] = obj_bbox
                labels[frame_ind, trackid]       = label 

            input_data.append(Image.open(os.path.join(base_path, frame_path)))

        vid_data, bbox_data = self.transforms(input_data, bbox_data)

        bbox_data = bbox_data.type(torch.LongTensor)
        xmin_data  = bbox_data[:,:,0]
        ymin_data  = bbox_data[:,:,1]
        xmax_data  = bbox_data[:,:,2]
        ymax_data  = bbox_data[:,:,3]
        labels     = torch.from_numpy(labels)

        # Permute the PIL dimensions (Frame, Height, Width, Chan) to pytorch (Chan, frame, height, width) 
        vid_data = vid_data.permute(3, 0, 1, 2)

        ret_dict = dict() 
        ret_dict['data']       = vid_data 
        ret_dict['xmin']       = xmin_data
        ret_dict['ymin']       = ymin_data
        ret_dict['xmax']       = xmax_data
        ret_dict['ymax']       = ymax_data
        ret_dict['bbox_data']  = bbox_data
        ret_dict['labels']     = labels

        return ret_dict

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
        elif crop_type == 'Center':
            self.transforms.append(pt.CenterCropClip(*crop_shape))

        self.transforms.append(pt.ResizeClip(*resize_shape))
        #self.transforms.append(pt.RandomFlipClip(direction='h', p=1.0))
        #self.transforms.append(pt.RandomRotateClip())
        self.transforms.append(pt.ToTensorClip())

    def __call__(self, input_data, bbox_data):
        """
        Preprocess the clip and the bbox data accordingly
        Args:
            input_data: List of PIL images containing clip frames 
            bbox_data:  Numpy array containing bbox coordinates per object per frame 

        Return:
            input_data: Pytorch tensor containing the processed clip data 
            bbox_data:  Numpy tensor containing the augmented bbox coordinates
        """
        for transform in self.transforms:
            input_data, bbox_data = transform(input_data, bbox_data)
            
        return input_data, bbox_data

class PreprocessEval(object):
    """
    Container for all transforms used to preprocess clips for evaluation in this dataset.
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
        self.transforms.append(pt.ToTensorClip())



    def __call__(self, input_data, bbox_data):
        """
        Preprocess the clip and the bbox data accordingly
        Args:
            input_data: List of PIL images containing clip frames 
            bbox_data:  Numpy array containing bbox coordinates per object per frame 

        Return:
            input_data: Pytorch tensor containing the processed clip data 
            bbox_data:  Numpy tensor containing the augmented bbox coordinates
        """
        for transform in self.transforms:
            input_data, bbox_data = transform(input_data, bbox_data)

        return input_data, bbox_data




