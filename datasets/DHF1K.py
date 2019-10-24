import torch
import torchvision
from .abstract_datasets import DetectionDataset 
import cv2
import os
import numpy as np
import json
import datasets.preprocessing_transforms as pt

class DHF1K(DetectionDataset):
    def __init__(self, *args, **kwargs):
        super(DHF1K, self).__init__(*args, **kwargs)

        # Get model object in case preprocessing other than default is used
        self.model_object   = kwargs['model_obj']
        self.load_type = kwargs['load_type']
        
        self.model_output_shape = 0 
        if 'model_output_shape' in kwargs.keys():
            self.model_output_shape = kwargs['model_output_shape']
        
        print(self.load_type)
        if self.load_type=='train':
            self.transforms = kwargs['model_obj'].train_transforms
        
        else:
            self.transforms = kwargs['model_obj'].test_transforms
    
        if 'specific_vid' in kwargs.keys():
            self.specific_vid = kwargs['specific_vid']
            self.vid_indices = np.where(np.array([self.samples[i]['base_path'] for i in range(len(self.samples))])==self.specific_vid)[0]
            self.specific_vid_idx = 0
        else:
            self.specific_vid = 0

    
    def __getitem__(self, idx):
        if self.specific_vid != 0:
            idx = self.vid_indices[self.specific_vid_idx]
            self.specific_vid_idx+=1

        vid_info = self.samples[idx]

        
        base_path = vid_info['base_path']
        vid_size  = vid_info['frame_size']

        input_data = []
        map_data = []
        bin_data = []

        for frame_ind in range(len(vid_info['frames'])):
            frame      = vid_info['frames'][frame_ind]
            frame_path = frame['img_path']
            map_path   = frame['map_path']
            bin_path   = frame['bin_path']
            
            # Load frame, convert to RGB from BGR and normalize from 0 to 1
            #input_data.append(cv2.imread(os.path.join('/z/home/erichof/ViP_Git/tasednet_test/TASED-Net/example/ex1', str(frame_ind+1).zfill(4)+'.jpg'))[...,::-1]/255.)
            input_data.append(cv2.imread(os.path.join(base_path, frame_path))[...,::-1]/255.)
            
            # Load frame, Normalize from 0 to 1
            # All frame channels have repeated values
            map_data.append(cv2.imread(map_path)/255.)
            bin_data.append(cv2.imread(bin_path)/255.)


        #targets = data['map']
        #bin_targets = data['bin']
        #
        #ts = targets.shape
        #ps = predictions.shape
        #targets = np.array(targets[:,:,int(ts[2]/2),:,:])
        #bin_targets = np.array(bin_targets[:,:,int(ts[2]/2),:,:])
        #res = np.zeros((ts[0], ts[1], ps[2], ps[3])) 
        #res_bin = np.zeros((ts[0], ts[1], ps[2], ps[3])) 
        #for i in range(ts[0]):
        #    res[i,0] = cv2.resize(targets[i,0], (ps[2], ps[3]), interpolation=cv2.INTER_AREA)
        #    res_bin[i,0] = cv2.resize(bin_targets[i,0], (ps[2], ps[3]), interpolation=cv2.INTER_AREA)
        #    
        #targets = np.clip(res, a_min=0, a_max=None)
        #bin_targets = np.clip(self.discretize_gt(res_bin), a_min=0, a_max=None)




        #temp_vid_data = self.transforms(input_data)
        vid_data = self.transforms(input_data)

        #bin_data = temp_vid_data[2::3]  # Binary locations of human gaze
        #map_data = temp_vid_data[1::3]  # Gaussian applied to binary locations of human gaze, real valued
        #vid_data = temp_vid_data[::3]   # Video pixel data

        
        #if self.model_output_shape != 0:
            #shape = map_data.shape
            #map_data = map_data.numpy()
            #bin_data = bin_data.numpy()
            #map_data_res = np.zeros([shape[0], self.model_output_shape[0], self.model_output_shape[1], shape[3]])
            #bin_data_res = np.zeros([shape[0], self.model_output_shape[0], self.model_output_shape[1], shape[3]])
            #for frame_ind in range(shape[0]):
            #    map_data_res[frame_ind] = cv2.resize(map_data[frame_ind], (self.model_output_shape[0], self.model_output_shape[1]) , interpolation=cv2.INTER_AREA) 
            #    bin_data_res[frame_ind] = cv2.resize(bin_data[frame_ind], (self.model_output_shape[0], self.model_output_shape[1]), interpolation=cv2.INTER_AREA) 

            #map_data_res = np.clip(map_data_res, a_min=0, a_max=None)
            #bin_data_res = np.clip(np.where(bin_data_res > 0.005, 1, 0), a_min=0, a_max=None)

            #map_data = torch.Tensor(map_data_res)
            #bin_data = torch.Tensor(bin_data_res)
        map_data = torch.Tensor(map_data)
        bin_data = torch.Tensor(bin_data)

        # Permute the PIL dimensions (Frame, Height, Width, Chan) to pytorch (Chan, frame, height, width) 
        vid_data = vid_data.permute(3, 0, 1, 2)
        map_data = map_data.permute(3, 0, 1, 2)
        bin_data = bin_data.permute(3, 0, 1, 2)
        # All channels are repeated so remove the unnecessary channels
        map_data = map_data[0].unsqueeze(0)
        bin_data = bin_data[0].unsqueeze(0)


        ret_dict         = dict() 
        ret_dict['data'] = vid_data 

        annot_dict                = dict()
        annot_dict['data']        = vid_data
        annot_dict['map']         = map_data
        annot_dict['bin']         = bin_data
        annot_dict['input_shape'] = vid_data.size()
        annot_dict['name']        = base_path
        ret_dict['annots']        = annot_dict

        return ret_dict
