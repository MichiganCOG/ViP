import torch
from .abstract_datasets import RecognitionDataset 
from PIL import Image
import cv2
import os
import numpy as np
import datasets.preprocessing_transforms as pt
from torchvision import transforms

class UCF101(RecognitionDataset):
    def __init__(self, *args, **kwargs):
        """
        Initialize UCF101 class  
        Args:
            load_type    (String): Select training or testing set 
            resize_shape (Int):    [Int, Int] Array indicating desired height and width to resize input
            crop_shape   (Int):    [Int, Int] Array indicating desired height and width to crop input
            final_shape  (Int):    [Int, Int] Array indicating desired height and width of input to deep network
            preprocess   (String): Keyword to select different preprocessing types            

        Return:
            None
        """
        super(UCF101, self).__init__(*args, **kwargs)

        self.load_type    = kwargs['load_type']
        self.resize_shape = kwargs['resize_shape']
        self.crop_shape   = kwargs['crop_shape']
        self.final_shape  = kwargs['final_shape']
        self.preprocess   = kwargs['preprocess']
        
        if self.load_type=='train':
            self.transforms = kwargs['model_obj'].train_transforms 

        else:
            self.transforms = kwargs['model_obj'].test_transforms


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
            try:
                # Load frame image data and preprocess image accordingly
                input_data.append(cv2.imread(frame_path)[...,::-1]/1.)

            except:
                print(frame_path)


        # Preprocess data
        vid_data   = self.transforms(input_data)
        labels     = torch.from_numpy(labels).float()

        # Permute the PIL dimensions (Frame, Height, Width, Chan) to pytorch (Chan, frame, height, width) 
        vid_data = vid_data.permute(3, 0, 1, 2)

        ret_dict           = dict() 
        ret_dict['data']   = vid_data 

        annot_dict           = dict()
        annot_dict['labels'] = labels

        ret_dict['annots']   = annot_dict

        return ret_dict


#dataset = HMDB51(json_path='/z/dat/HMDB51', dataset_type='train', clip_length=100, num_clips=0)
#dat = dataset.__getitem__(0)
#import pdb; pdb.set_trace()
