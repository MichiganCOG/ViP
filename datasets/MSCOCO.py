import torch
from .abstract_datasets import DetectionDataset 
import cv2
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
            self.transforms = kwargs['model_obj'].train_transforms 

        else:
            self.transforms = kwargs['model_obj'].test_transforms


    def __getitem__(self, idx):
        vid_info = self.samples[idx]
        
        base_path = vid_info['base_path']
        vid_size  = vid_info['frame_size']

        input_data = []
        vid_length = len(vid_info['frames'])
        vid_data   = np.zeros((vid_length, self.final_shape[0], self.final_shape[1], 3))-1
        bbox_data  = np.zeros((vid_length, self.max_objects, 4))-1
        labels     = np.zeros((vid_length, self.max_objects))-1
        iscrowds   = np.zeros((vid_length, self.max_objects))-1




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


            input_data.append(cv2.imread(os.path.join(base_path, frame_path))[...,::-1])

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

        ret_dict   = dict() 
        annot_dict = dict()
        annot_dict['xmin']       = xmin_data
        annot_dict['ymin']       = ymin_data
        annot_dict['xmax']       = xmax_data
        annot_dict['ymax']       = ymax_data
        annot_dict['bbox_data']  = bbox_data
        annot_dict['labels']     = labels
        annot_dict['iscrowds']   = iscrowds
        ret_dict['annots']       = annot_dict
        ret_dict['data']         = vid_data 

        return ret_dict
