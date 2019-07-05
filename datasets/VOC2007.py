import torch
from .abstract_datasets import DetectionDataset 
from PIL import Image
import cv2
import os
import numpy as np
import datasets.preprocessing_transforms as pt

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
            self.transforms = PreprocessTrain(**kwargs)

        else:
            self.transforms = PreprocessEval(**kwargs)


    def __getitem__(self, idx):
        vid_info = self.samples[idx]
        
        base_path = vid_info['base_path']
        vid_size  = vid_info['frame_size']

        input_data  = []
        vid_data    = np.zeros((self.clip_length, self.final_shape[0], self.final_shape[1], 3))-1
        bbox_data   = np.zeros((self.clip_length, self.max_objects, 4))-1
        labels      = np.zeros((self.clip_length, self.max_objects))-1
        diff_labels = np.zeros((self.clip_length, self.max_objects)) #difficult object labels

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

            input_data.append(cv2.imread(os.path.join(base_path, frame_path), cv2.IMREAD_COLOR))

        #vid_data, bbox_data = self.transforms(input_data, bbox_data)
        ####
        vid_data = torch.zeros((1,300,300,3))
        mean = np.array((104, 117, 123), dtype=np.float32) #assuming BGR channel order

        for i,pil_img in enumerate(input_data):
            cv2_img = np.array(pil_img)

            x = cv2.resize(cv2_img, (300, 300)).astype(np.float32)
            x -= mean
            x = x.astype(np.float32)[:, :, (2,1,0)] #BGR to RGB

            vid_data[i] = torch.Tensor(x)
        
        bbox_data = torch.Tensor(bbox_data)

        ####

        bbox_data   = bbox_data.type(torch.LongTensor)
        xmin_data   = bbox_data[:,:,0]
        ymin_data   = bbox_data[:,:,1]
        xmax_data   = bbox_data[:,:,2]
        ymax_data   = bbox_data[:,:,3]
        labels      = torch.from_numpy(labels).type(torch.LongTensor)
        diff_labels = torch.from_numpy(diff_labels).type(torch.LongTensor)

        #mean = torch.Tensor([[[[104, 117, 123]]]]) 
        #vid_data = vid_data - mean
        # Permute the vid dimensions (Frame, Height, Width, Chan) to PyTorch (Chan, Frame, Height, Width) 
        vid_data = vid_data.permute(3, 0, 1, 2)

        ret_dict = dict() 
        ret_dict['data']       = vid_data 
        ret_dict['height']     = torch.Tensor([vid_size[0]])
        ret_dict['width']     = torch.Tensor([vid_size[1]])
        ret_dict['xmin']       = xmin_data
        ret_dict['ymin']       = ymin_data
        ret_dict['xmax']       = xmax_data
        ret_dict['ymax']       = ymax_data
        ret_dict['bbox_data']  = bbox_data
        ret_dict['labels']     = torch.cat((bbox_data, labels.unsqueeze(2)),2) #[xmin,ymin,xmax,ymax,class_id]
        ret_dict['diff_labels'] = diff_labels 

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
        elif crop_type == 'Center':
            self.transforms.append(pt.CenterCropClip(*crop_shape))

        self.transforms.append(pt.ResizeClip(**kwargs))
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




