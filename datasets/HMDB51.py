import torch

from .abstract_datasets import RecognitionDataset 
from PIL import Image
import os
import numpy as np

class HMDB51(RecognitionDataset):
    def __init__(self, *args, **kwargs):
        super(HMDB51, self).__init__(*args, **kwargs)


    def __getitem__(self, idx):
        vid_info  = self.samples[idx]
        base_path = vid_info['base_path']

        vid_data   = np.zeros((self.clip_length, self.final_shape[0], self.final_shape[1], 3))-1
        labels     = np.zeros((self.clip_length))-1
    
        for frame_ind in range(len(vid_info['frames'])):
            frame_path   = os.path.join(base_path, vid_info['frames'][frame_ind]['img_path'])

            for frame_labels in vid_info['frames'][frame_ind]['actions']:
                labels[frame_ind] = frame_labels['action_class']

            # Load frame image data and preprocess image accordingly
            #vid_data[frame_ind] = self._preprocFrame(frame_path)

        vid_data   = torch.from_numpy(vid_data).float()
        labels     = torch.from_numpy(labels).float()

        # Permute the PIL dimensions (Frame, Height, Width, Chan) to pytorch (Chan, frame, height, width) 
        vid_data = vid_data.permute(3, 0, 1, 2)

        ret_dict           = dict() 
        ret_dict['data']   = vid_data 
        ret_dict['labels'] = labels


        return ret_dict

#dataset = HMDB51(json_path='/z/dat/HMDB51', dataset_type='train', clip_length=100, num_clips=0)
#dat = dataset.__getitem__(0)
#import pdb; pdb.set_trace()
