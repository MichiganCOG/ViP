from abc import ABCMeta
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image

class VideoDataset(Dataset):
    __metaclass__ = ABCMeta
    def __init__(self, json_path, dataset_type, clip_length=16, clip_offset=0, clip_stride=1, num_clips=-1, resize_shape=[128, 128], crop_shape=[128, 128], crop_type='random', final_shape=[128,128]):
        """
        Args: 
            json_path:    Path to the directory containing the dataset's JSON file (not including the file itself)
            dataset_type: String indicating whether to load training or validation data ('train' or 'val') 
            clip_length:  Number of frames in each clip that will be input into the network
            clip_offset:  Number of frames from beginning of video to start extracting clips 
            clip_stride:  The temporal stride between clips
            num_clips:    Number of clips to extract from each video (-1 uses the entire video)
            resize_shape: The shape [h, w] of each frame after resizing
            crop_shape:   The shape [h, w] of each frame after cropping
            crop_type:    The method used to crop (either random or center)
            final_shape:  Final shape [h, w] of each frame after all preprocessing, this is input to network
        """
        # JSON loading arguments
        self.json_path      = json_path 
        self.dataset_type   = dataset_type
        
        # Clips processing arguments
        self.clip_length    = clip_length 
        self.clip_offset    = clip_offset
        self.clip_stride    = clip_stride
        self.num_clips      = num_clips

        # Frame-wise processing arguments
        self.resize_shape   = resize_shape 
        self.crop_shape     = crop_shape 
        self.crop_type      = crop_type 
        self.final_shape    = final_shape

        # Creates the self.samples list which will be indexed by each __getitem__ call
        self._getClips()


    def __getitem__(self, idx):
        raise NotImplementedError("Dataset must contain __getitem__ method which loads videos from memory.")

    def _getClips(self):
        """
        Loads the JSON file associated with the videos in a datasets and processes each of them into clips
        """
        raise NotImplementedError("Dataset must contain getClips method which loads and processes the dataset's JSON file.") 

    def _processClips(self, video):
        """
        Processes a single video into uniform sized clips that will be loaded by __getitem__
        Args:
            video:       List containing a dictionary of annontations per frame

        Additional Parameters:
            self.clip_length: Number of frames extracted from each clip
            self.num_clips:   Number of clips to extract from each video (-1 uses the entire video)
            self.clip_offset: Number of frames from beginning of video to start extracting clips 
            self.clip_stride: Number of frames between clips when extracting them from videos 
        """
        return [video[:self.clip_length]]

    def _preprocFrame(self, frame_path, bbox_data=[]):
        """
        Preprocess a frame using data augmentation functions and return processed image 
        Args:
            frame_path: The global path to the frame image 
            bbox_data:  For detection datasets, augment bbox coordinates according to frame augmentations (size = [# objects, 4 coordinates])
        """
        frame_data = Image.open(frame_path)
        frame_data = frame_data.resize(self.final_shape)
        # TODO @preetsg add calls to preproc_utils.py here

        if bbox_data==[]:
            return np.array(frame_data)

        else:
            return np.array(frame_data), bbox_data
        


class RecognitionDataset(VideoDataset):
    __metaclass__ = ABCMeta
    def __init__(self, *args, **kwargs):
        super(RecognitionDataset, self).__init__(*args, **kwargs)

    def _getClips(self):
        """
        Required format for all recognition dataset JSON files:
        
        List(Vidnumber: Dict{
                   List(Frames: Dict{
                                     Frame Size,
                                     Frame Path,
                                     List(Actions: Dict{
                                                        Track ID
                                                        Action Class
                                     }) End Object List in Frame
        
                   }) End Frame List in Video
        
                   Str(Base Vid Path)
        
             }) End Video List in Dataset
        
        Eg: action_label = dataset[vid_index]['frames'][frame_index]['actoins'][action_index]['action_class']
        """

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
        
   
        base_path = os.path.join(data_path, self.dataset_type+'images')
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




class DetectionDataset(VideoDataset):
    __metaclass__ = ABCMeta
    def __init__(self, *args, **kwargs):
        super(DetectionDataset, self).__init__(*args, **kwargs)

    def _getClips(self):
        """
        Required format for all detection datset JSON files:
        
        List(Vidnumber: Dict{
                   List(Frames 'frame': Dict{
                                     Frame Size 'frame_size',
                                     Frame Path 'img_path',
                                     List(Objects 'objs': Dict{
                                                        Track ID 'trackid'
                                                        Object Class 'c'
                                                        Occluded 'occ'
                                                        List(Bounding box coordinates 'bbox': [xmin, ymin, xmax, ymax])
        
                                     }) End Object List in Frame
        
                   }) End Frame List in Video
        
                   Str(Base Vid Path) 'base_path
        
             }) End Video List in Dataset
        
        Ex: coordinates = dataset[vid_index]['frame'][frame_index]['objs'][obj_index]['bbox']
        """

        # Load all video paths into the samples array to be loaded by __getitem__ 

        self.samples = []
        self.dataset_type = 'train'
        
        if self.dataset_type == 'train':
            full_json_path = os.path.join(self.json_path, 'train.json')

        else:
            full_json_path = os.path.join(self.json_path, 'val.json') 

        json_file = open(full_json_path,'r')
        json_data = json.load(json_file) 
        json_file.close()

        # Load the information for each video and process it into clips
        for video_info in json_data:
            clips = self._processClips(video_info['frame'])

            # Each clip is a list of dictionaries per frame containing information
            # Example info: object bbox annotations, object classes, frame img path
            for clip in clips:    
                self.samples.append(dict(frames=clip, base_path=video_info['base_path'], frame_size=video_info['frame_size']))


