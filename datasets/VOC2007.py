import torch
from .abstract_datasets import DetectionDataset 
from PIL import Image
import cv2
import os
import numpy as np

class VOC2007(DetectionDataset):
    def __init__(self, *args, **kwargs):
        super(VOC2007, self).__init__(*args, **kwargs)

        self.load_type = kwargs['load_type']

        # Maximum number of annotated object present in a single frame in entire dataset
        # Dictates the return size of annotations in __getitem__
        self.max_objects = 50 #TODO: Verify real value

        #Map class name to a class id
        self.class_to_id = {'aeroplane':0,
                            'bicycle':1,
                            'bird':2,
                            'boat':3,
                            'bottle':4,
                            'bus':5,
                            'car':6,
                            'cat':7,
                            'chair':8,
                            'cow':9,
                            'diningtable':10,
                            'dog':11,
                            'horse':12,
                            'motorbike':13,
                            'person':14,
                            'pottedplant':15,
                            'sheep':16,
                            'sofa':17,
                            'train':18,
                            'tvmonitor':19
                            }
        #TODO: maybe add a reverse mapping

        if self.load_type=='train':
            self.transforms = kwargs['model_obj'].train_transforms

        else:
            self.transforms = kwargs['model_obj'].test_transforms


    def __getitem__(self, idx):
        vid_info = self.samples[idx]
        
        base_path = vid_info['base_path']
        vid_size  = vid_info['frame_size']

        input_data  = []
        vid_length = len(vid_info['frames'])
        vid_data    = np.zeros((vid_length, self.final_shape[0], self.final_shape[1], 3))-1
        bbox_data   = np.zeros((vid_length, self.max_objects, 4))-1
        labels      = np.zeros((vid_length, self.max_objects))-1
        diff_labels = np.zeros((vid_length, self.max_objects)) #difficult object labels

        for frame_ind in range(len(vid_info['frames'])):
            frame      = vid_info['frames'][frame_ind]
            frame_path = frame['img_path']
            
            # Extract bbox and label data from video info
            for obj in frame['objs']:
                trackid   = obj['trackid']
                label     = self.class_to_id[obj['c']]
                obj_bbox  = obj['bbox'] # [xmin, ymin, xmax, ymax]
                difficult = obj['difficult']
                
                bbox_data[frame_ind, trackid, :] = obj_bbox
                labels[frame_ind, trackid]       = label 
                diff_labels[frame_ind, trackid]  = difficult 

            #Read each image and change from BGR to RGB
            input_data.append(cv2.imread(os.path.join(base_path, frame_path), cv2.IMREAD_COLOR)[:,:,(2,1,0)])

        vid_data, bbox_data = self.transforms(input_data, bbox_data) #preprocess frames

        bbox_data   = bbox_data.type(torch.LongTensor)
        xmin_data   = bbox_data[:,:,0]
        ymin_data   = bbox_data[:,:,1]
        xmax_data   = bbox_data[:,:,2]
        ymax_data   = bbox_data[:,:,3]
        labels      = torch.from_numpy(labels).type(torch.LongTensor)
        diff_labels = torch.from_numpy(diff_labels).type(torch.LongTensor)

        # Permute the vid dimensions (Frame, Height, Width, Chan) to PyTorch (Chan, Frame, Height, Width) 
        vid_data = vid_data.permute(3, 0, 1, 2)

        ret_dict = dict() 
        ret_dict['data']       = vid_data 

        annot_dict = dict()
        annot_dict['height']     = torch.Tensor([vid_size[0]])
        annot_dict['width']      = torch.Tensor([vid_size[1]])
        annot_dict['xmin']       = xmin_data
        annot_dict['ymin']       = ymin_data
        annot_dict['xmax']       = xmax_data
        annot_dict['ymax']       = ymax_data
        annot_dict['bbox_data']  = bbox_data
        annot_dict['labels']     = torch.cat((bbox_data, labels.unsqueeze(2)),2) #[xmin,ymin,xmax,ymax,class_id]
        annot_dict['diff_labels'] = diff_labels 
        ret_dict['annots'] = annot_dict

        return ret_dict
