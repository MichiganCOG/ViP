from abc import ABCMeta
from torch.utils.data import Dataset
import json
import numpy as np
import os

class VideoDataset(Dataset):
    __metaclass__ = ABCMeta
    def __init__(self, seq_length, resize_shape, crop_shape, crop_type='random', strides=1):
        """
        Args: 
            seq_length:   Number of frames in each clip that will be input into the network
            resize_shape: The shape [h, w] of each frame after resizing
            crop_shape:   The shape [h, w] of each frame after cropping
            crop_type:    The method used to crop (either random or center)
            strides:      The temporal stride between clips
        """
        self.seq_length     = seq_length 
        self.resize_shape   = resize_shape 
        self.crop_shape     = crop_shape 
        self.crop_type      = crop_type 
        self.strides        = strides 

        self.getClips()

    def __getitem__(self, idx):
        raise NotImplementedError("Dataset must contain __getitem__ method which loads videos from memory.")

    def getClips(self):
        raise NotImplementedError("Dataset must contain getClips method which loads and processes the dataset's JSON file.") 
        


class RecognitionDataset(VideoDataset):
    __metaclass__ = ABCMeta
    def __init__(self, *args, **kwargs):
        super(RecognitionDataset, self).__init__(*args, **kwargs)

    def getClips(self):
        data_path  = os.path.join('/z/dat', self.dataset_dir)
        label_dict = {'brush_hair': 0,      'cartwheel': 1,
                      'catch': 2,           'chew': 3, 
                      'clap': 4,            'climb_stairs': 5, 
                      'climb': 6,           'dive': 7,
                      'draw_sword': 8,      'dribble': 9,
                      'drink': 10,          'eat': 11,
                      'fall_floor': 12,     'fencing': 13,
                      'flic_flac': 14,      'golf': 15,
                      'handstand': 16,      'hit': 17,
                      'hug': 18,            'jump': 19,
                      'kick_ball': 20,      'kick': 21,
                      'kiss': 22,           'laugh': 23,
                      'pick': 24,           'pour': 25,
                      'pullup': 26,         'punch': 27,
                      'push': 28,           'pushup': 29,
                      'ride_bike': 30,      'ride_horse': 31,
                      'run': 32,            'shake_hands': 33,
                      'shoot_ball': 34,     'shoot_bow': 35,
                      'shoot_gun': 36,      'sit': 37, 
                      'situp': 38,          'smile': 39,
                      'smoke': 40,          'somersault': 41,
                      'stand': 42,          'swing_baseball': 43,
                      'sword_exercise': 44, 'sword': 45,
                      'talk': 46,           'throw': 47,
                      'turn': 48,           'walk': 49,
                      'wave': 50}
        
   
        base_path = os.path.join(data_path, self.train_or_val+'images')
        actions   = os.listdir(base_path)
        
        for action in actions:
            for video in os.listdir(os.path.join(base_path, action)):
                if not '.DS' in video:
                    video_images = sorted(os.listdir(os.path.join(base_path, action, video.replace('.avi',''))))
                    self.samples.append([os.path.join(base_path, action, video.replace('.avi',''), video_image) for video_image in video_images])
                    self.labels.append(label_dict[action])
            
                # END IF

            # END FOR

        # END FOR

        # TODO Implement JSON reader
        return []




class DetectionDataset(VideoDataset):
    __metaclass__ = ABCMeta
    def __init__(self, *args, **kwargs):
        super(DetectionDataset, self).__init__(*args, **kwargs)

    def getClips(self):
        #json_train_path = '/z/home/natlouis/pytorch_goturn/data/ILSVRC2015_VID/ilsvrc_train.json'
        #json_train_file = open(json_train_path,'r')
        #json_train_data = json.load(json_train_file) 
        #json_train_file.close()

        #json_val_path = '/z/home/natlouis/pytorch_goturn/data/ILSVRC2015_VID/ilsvrc_val.json'
        #json_val_file = open(json_val_path,'r')
        #json_val_data = json.load(json_val_file) 
        #json_val_file.close()
        #
        #json_snippet_train_path = '/z/home/natlouis/pytorch_goturn/data/ILSVRC2015_VID/ilsvrc_train_snippet.json'
        #json_snippet_train_file = open(json_snippet_train_path,'r')
        #json_snippet_train_data = json.load(json_snippet_train_file) 
        #json_snippet_train_file.close()

        #json_snippet_val_path = '/z/home/natlouis/pytorch_goturn/data/ILSVRC2015_VID/ilsvrc_val_snippet.json'
        #json_snippet_val_file = open(json_snippet_val_path,'r')
        #json_snippet_val_data = json.load(json_snippet_val_file) 
        #json_snippet_val_file.close()

        #self.train_samples = []
        #self.train_labels  = []
        #for vid in json_snippet_train_data:
        #    self.train_samples.append(

        #self.labels_dict = np.load('/z/home/erichof/datasets/ILSVRC2015/labels_number_keys.npy').tolist()
        #self.label_values = self.labels_dict.values()
        #self.label_values.sort()



        # Load all video paths into the samples array to be loaded by __getitem__ 

        self.samples = []
        self.dataset_dir = '/z/home/natlouis/data/ILSVRC2015/'
        self.train_or_val = 'train'
        
        annot_path = os.path.join(self.dataset_dir, 'Annotations/VID/')
        #data_path = os.path.join(self.dataset_dir, 'Data/VID/')

        
        if self.train_or_val == 'val':
            self.samples = os.listdir(os.path.join(annot_path, self.train_or_val))
            self.samples.sort()

        else:
            self.samples = []
            for vid_dir in os.listdir(os.path.join(annot_path, self.train_or_val)):
                self.samples = self.samples + [ vid_dir + '/' + s for s in os.listdir(os.path.join(annot_path, self.train_or_val, vid_dir))]



