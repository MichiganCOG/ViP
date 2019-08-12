import torch
import torchvision

from .abstract_datasets import DetectionDataset 
#OR
from .abstract_datasets import RecognitionDataset 

import cv2
import os
import numpy as np
import datasets.preprocessing_transforms as pt


class CustomDataset(DetectionDataset):
    # OR
class CustomDataset(RecognitionDataset):
    def __init__(self, *args, **kwargs):
        super(CustomDataset, self).__init__(*args, **kwargs)

        # Get model object in case preprocessing other than default is used
        self.model_object   = kwargs['model_object']
        self.load_type = kwargs['load_type']

        if self.load_type=='train':
            self.transforms = kwargs['model_obj'].train_transforms

        else:
            self.transforms = kwargs['model_obj'].test_transforms

    def __getitem__(self, idx):
        vid_info = self.samples[idx]
        
        base_path = vid_info['base_path']
        vid_size  = vid_info['frame_size']

        input_data = []
        vid_data   = np.zeros((self.clip_length, self.final_shape[0], self.final_shape[1], 3))-1
        labels     = np.zeros((self.clip_length, self.max_objects))-1

        # For Detection Datasets
        bbox_data  = np.zeros((self.clip_length, self.max_objects, 4))-1




        for frame_ind in range(len(vid_info['frames'])):
            frame      = vid_info['frames'][frame_ind]
            frame_path = frame['img_path']
            
            # Extract bbox and label data from video info
            for obj in frame['objs']:
                trackid   = obj['trackid']
                label     = obj['label']
                obj_bbox  = obj['bbox'] # [xmin, ymin, xmax, ymax]

                bbox_data[frame_ind, trackid, :] = obj_bbox
                labels[frame_ind, trackid]       = label 
            
            # Read framewise image data from storage
            input_data.append(cv2.imread(os.path.join(base_path, frame_path))[...,::-1]/255.)

        # Apply preprocessing transformations
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
        annot_dict = dict()
        annot_dict['xmin']        = xmin_data
        annot_dict['ymin']        = ymin_data
        annot_dict['xmax']        = xmax_data
        annot_dict['ymax']        = ymax_data
        annot_dict['bbox_data']   = bbox_data
        annot_dict['labels']      = labels
        annot_dict['input_shape'] = vid_data.size() 
        ret_dict['annots']     = annot_dict

        return ret_dict
