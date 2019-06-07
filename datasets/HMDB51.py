import torch
from abstract_datasets import DetectionDataset 
from PIL import Image
import os
import numpy as np

class HMDB51(RecognitionDataset):
    def __init__(self, *args, **kwargs):
        super(HMDB51, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        cat             = None 
        paths           = self.samples[idx]


        if self.train_or_val == 'train':
            flip_prop = np.random.randint(low=0, high=2)

        else:
            flip_prop = 1

        # END IF

        # Uniform Sampling
        if len(paths) < self.seq_length:
            indices = np.ceil(self.seq_length/float(len(paths)))
            indices = indices.astype('int32')
            indices = np.tile(np.arange(0, len(paths), 1, dtype='int32'), indices)
            indices = indices[np.linspace(0, len(indices)-1, self.seq_length, dtype='int32')]

        else:
            indices = np.linspace(0, len(paths)-1, self.seq_length, dtype='int32')

        # END IF

        ## Continuous Frames
        #if len(paths) < self.seq_length:
        #    indices = np.ceil(self.seq_length/float(len(paths)))
        #    indices = indices.astype('int32')
        #    indices = np.tile(np.arange(0, len(paths), 1, dtype='int32'), indices)
        #    indices = indices[:self.seq_length]

        #elif len(paths) == self.seq_length:
        #    indices = np.arange(0, self.seq_length, 1, dtype='int32')

        #else:
        #    indices = np.random.randint(0, len(paths)-self.seq_length, 1)
        #    indices = np.arange(indices, indices+self.seq_length, 1, dtype='int32')

        ## END IF
     
        for step_idx, _idx in enumerate(indices):
            image_path = paths[_idx]
            image      = np.array(cv2.imread(image_path)).astype(np.float64)

            ######### Preprocessing Steps #######################
            # Resize image
            if self.resize_shape is not None:
                resizer = partial(cv2.resize, dsize = tuple(self.resize_shape[::-1]))
                image = resizer(image)

            ## BGR since opencv read is used to load data
            #image -= np.array([[[90.0, 98.0, 102.0]]])
            #image -= self.mean[step_idx, :, :, :] 

            ## Randomly crop 
            #if self.crop_type == 'random':
            #    image = get_random_crop(image, self.crop_shape[0], self.crop_shape[1])

            #if self.crop_type == 'center': 
            #    image = get_center_crop(image, self.crop_shape[0], self.crop_shape[1])

            ## Randomly horizontally flip 
            #if flip_prop == 0:
            #    image = cv2.flip(image, 1)


            ## Change to RGB for network input
            #image = image[...,::-1].copy() 

            ######### Preprocessing Steps #######################
            if cat is None:
                cat = np.expand_dims(image, axis=0)

            else:
                cat = np.vstack((cat, np.expand_dims(image, axis=0)))

            # END IF

        # END FOR

        cat = np.transpose(cat, (3, 0, 1, 2))

        ret_data                = {}
        ret_data['data']        = torch.Tensor(cat)
        ret_data['true_labels'] = self.labels[idx]


        return ret_data
