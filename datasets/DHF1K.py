import torch
try:
    from .abstract_datasets import DetectionDataset 
except:
    from abstract_datasets import DetectionDataset 
import cv2
import os
import numpy as np
import json
try:
    import datasets.preprocessing_transforms as pt
except:
    import preprocessing_transforms as pt

class DHF1K(DetectionDataset):
    def __init__(self, *args, **kwargs):
        super(DHF1K, self).__init__(*args, **kwargs)

        # Get model object in case preprocessing other than default is used
        self.model_object   = kwargs['model_obj']
        self.load_type = kwargs['load_type']
        
        print(self.load_type)
        if self.load_type=='train':
            self.transforms = kwargs['model_obj'].train_transforms
        
        else:
            self.transforms = kwargs['model_obj'].test_transforms
    


    
    def __getitem__(self, idx):
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
            input_data.append(cv2.imread(os.path.join(base_path, frame_path))[...,::-1]/255.)
            
            # Load frame, Normalize from 0 to 1
            # All frame channels have repeated values
            map_data.append(cv2.imread(map_path)/255.)
            bin_data.append(cv2.imread(bin_path)/255.)



        vid_data = self.transforms(input_data)

        # Annotations must be resized in the loss/metric
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
        annot_dict['map']         = map_data
        annot_dict['bin']         = bin_data
        annot_dict['input_shape'] = vid_data.size()
        annot_dict['name']        = base_path
        ret_dict['annots']        = annot_dict

        return ret_dict


if __name__=='__main__':

    class tts():
        def __call__(self, x):
            return pt.ToTensorClip()(x)
    class debug_model():
        def __init__(self):
            self.train_transforms = tts()
    dataset = DHF1K(model_obj=debug_model(), json_path='/z/home/erichof/datasets/DHF1K', load_type='train', clip_length=16, clip_offset=0, clip_stride=1, num_clips=0, random_offset=0, resize_shape=0, crop_shape=0, crop_type='Center', final_shape=0, batch_size=1)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)


    import matplotlib.pyplot as plt
    for x in enumerate(train_loader):
        plt.imshow(x[1]['data'][0,:,0].permute(1,2,0).numpy())
        #plt.show()
        import pdb; pdb.set_trace()
