import torch
import torchvision
from .abstract_datasets import DetectionDataset 
import cv2
import os
import numpy as np
import json

class Manual_Hands(DetectionDataset):
    """
    Manually-annotated keypoints on hands for pose estimation.
    Includes images from The MPII Human Pose and New Zealand Sign Language (NZSL) datasets

    Source: https://arxiv.org/1704.07809
    """
    def __init__(self, *args, **kwargs):
        super(Manual_Hands, self).__init__(*args, **kwargs)

        self.load_type = kwargs['load_type']
        self.json_path = kwargs['json_path']

        # Maximum number of annotated object present in a single frame in entire dataset
        # Dictates the return size of annotations in __getitem__
        self.max_objects = 1
        self.sigma       = 3.0
        self.stride      = 8 #effective stride of the entire network

        if self.load_type=='train':
            self.transforms = kwargs['model_obj'].train_transforms

        else:
            self.transforms = kwargs['model_obj'].test_transforms

    #Adapted from: https://github.com/namedBen/Convolutional-Pose-Machines-Pytorch
    def gaussian_kernel(self, size_w, size_h, center_x, center_y, sigma):
        #Outputs a gaussian heat map on defined point
        gridy, gridx = torch.meshgrid(torch.arange(0,size_h,dtype=torch.float), torch.arange(0,size_w,dtype=torch.float))
        D2 = (gridx - center_x)**2 + (gridy - center_y)**2

        return torch.exp(-0.5 * D2 / sigma**2)

    def __getitem__(self, idx):
        vid_info = self.samples[idx]
        
        base_path = vid_info['base_path']
        vid_size  = vid_info['frame_size']

        input_data    = []

        vid_length = len(vid_info['frames'])
        vid_data      = np.zeros((vid_length, self.final_shape[0], self.final_shape[1], 3))-1
        bbox_data     = np.zeros((vid_length, self.max_objects, 4))-1
        hand_pts_data = np.zeros((vid_length, self.max_objects, 21, 3))-1
        labels        = np.zeros((vid_length, self.max_objects))-1
        occlusions    = np.zeros((vid_length, self.max_objects, 22), dtype=np.int32)-1 #21 keypoints + background = 22 points

        for frame_ind in range(len(vid_info['frames'])):
            frame          = vid_info['frames'][frame_ind]
            width, height  = vid_info['frame_size']
            frame_path     = frame['img_path']
            
            # Extract bbox and label data from video info
            for obj in frame['objs']:
                #trackid   = obj['trackid'] #Let's ignore trackid for now, only one annotation per image
                trackid   = 0
                label     = 1 if obj['c'] == 'left' else 0 #1: left hand, 0: right hand
                occluded  = obj['occ']
                obj_bbox  = obj['bbox'] # [xmin, ymin, xmax, ymax]
                body_pts  = obj['body_pts'] #16 points (x,y,valid)
                hand_pts  = obj['hand_pts'] #21 points (x,y,valid)
                head_box  = obj['head_box']
                head_size = obj['head_size'] #max dim of tightest box around head
                hand_ctr  = obj['hand_ctr']
                mpii      = obj['mpii']

                #During training square patch is 2.2*B where B is max(obj_bbox)
                if self.load_type == 'train':
                    B = max(obj_bbox[2]-obj_bbox[0], obj_bbox[3]-obj_bbox[1])
                else: #During testing B is 0.7*head_size
                    B = 0.7*head_size

                hand_size = 2.2*B
                xtl       = np.clip(int(hand_ctr[0]-hand_size/2), 0, width)
                ytl       = np.clip(int(hand_ctr[1]-hand_size/2), 0, height)
                xbr       = np.clip(int(hand_ctr[0]+hand_size/2), 0, width)
                ybr       = np.clip(int(hand_ctr[1]+hand_size/2), 0, height)

                hand_crop = [xtl, ytl, xbr, ybr]
                bbox_data[frame_ind, trackid, :]     = obj_bbox
                labels[frame_ind, trackid]           = label 
                hand_pts_data[frame_ind, trackid, :] = hand_pts
                occlusions[frame_ind, trackid]       = occluded + [0] #Add element for background

            # Load frame, convert to RGB from BGR and normalize from 0 to 1
            input_data.append(cv2.imread(os.path.join(base_path, frame_path))[...,::-1])

        #Crop hand and resize, perform same transforms to ground truth keypoints
        vid_data, hand_pts_coords = self.transforms(input_data, hand_pts_data[:,:,:,:2], hand_crop, labels)

        h_width  = int(self.final_shape[1]/self.stride)
        h_height = int(self.final_shape[0]/self.stride)
        heatmaps = torch.zeros((22, h_width, h_height), dtype=torch.float) #heatmaps for 21 keypoints + background
        for i,pts in enumerate(hand_pts_coords[0][0]):
            x = pts[0] / self.stride
            y = pts[1] / self.stride 
            heatmaps[i,:,:] = self.gaussian_kernel(h_width, h_height, x, y, self.sigma)

        heatmaps[-1,:,:] = 1 - torch.max(heatmaps[:-1,:,:], dim=0)[0] #Last layer is background

        vid_data = vid_data/255
        # Permute the PIL dimensions (Frame, Height, Width, Chan) to pytorch (Chan, frame, height, width) 
        vid_data = vid_data.permute(3, 0, 1, 2)
        vid_data = vid_data.squeeze(1) #Remove frame dimension, b/c this is an image dataset

        ret_dict = dict() 
        ret_dict['data']       = vid_data 
        annot_dict = dict()
        annot_dict['head_size']    = head_size
        annot_dict['hand_pts']    = hand_pts_coords 
        annot_dict['heatmaps']    = heatmaps
        annot_dict['labels']      = labels
        annot_dict['occ']         = occlusions 
        annot_dict['frame_path']  = frame_path 
        annot_dict['frame_size']  = vid_size #width, height
        ret_dict['annots']     = annot_dict

        return ret_dict
