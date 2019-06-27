import torch
from .abstract_datasets import DetectionDataset 
from PIL import Image
import os
import numpy as np
import datasets.preprocessing_transforms as pt

class MSCOCO(DetectionDataset):
    def __init__(self, *args, **kwargs):
        super(MSCOCO, self).__init__(*args, **kwargs)

        self.load_type = kwargs['load_type']

        # Some category labels are missing so the labels in the dataset must be indexed into the array below for the network to output consecutive labels
        self.category_remap = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


        # Maximum number of annotated object present in a single frame in entire dataset
        # Dictates the return size of annotations in __getitem__
        self.max_objects = 93


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
        iscrowds   = np.zeros((self.clip_length, self.max_objects))-1




        for frame_ind in range(len(vid_info['frames'])):
            frame      = vid_info['frames'][frame_ind]
            frame_path = frame['img_path']
            
            # Extract bbox and label data from video info
            for obj in frame['objs']:
                trackid   = obj['trackid']
                label     = obj['c']
                iscrowd   = obj['iscrowd']
                obj_bbox  = obj['bbox'] # [xmin, ymin, xmax, ymax]
                
                # Category remap
                label      = self.category_remap.index(label)


                bbox_data[frame_ind, trackid, :] = obj_bbox
                labels[frame_ind, trackid]       = label 
                iscrowds[frame_ind, trackid]     = iscrowd


            input_data.append(Image.open(os.path.join(base_path, frame_path)))

        vid_data, bbox_data = self.transforms(input_data, bbox_data)

        bbox_data = bbox_data.type(torch.LongTensor)
        xmin_data  = bbox_data[:,:,0]
        ymin_data  = bbox_data[:,:,1]
        xmax_data  = bbox_data[:,:,2]
        ymax_data  = bbox_data[:,:,3]
        labels     = torch.from_numpy(labels)
        iscrowds   = torch.from_numpy(iscrowds)

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
        ret_dict['iscrowds']   = iscrowds

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
        else:
            self.transforms.append(pt.CenterCropClip(*crop_shape))

        self.transforms.append(pt.ResizeClip(*resize_shape))
        self.transforms.append(pt.RandomFlipClip(direction='h', p=1.0))
        self.transforms.append(pt.RandomRotateClip())
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




