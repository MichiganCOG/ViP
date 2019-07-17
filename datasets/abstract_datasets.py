from abc import ABCMeta
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image

class VideoDataset(Dataset):
    __metaclass__ = ABCMeta
    def __init__(self, *args, **kwargs):
    #def __init__(self, json_path, load_type, clip_length=16, clip_offset=0, clip_stride=1, num_clips=-1, resize_shape=[128, 128], crop_shape=[128, 128], crop_type='random', final_shape=[128,128], *args, **kwargs):
        """
        Args: 
            json_path:    Path to the directory containing the dataset's JSON file (not including the file itself)
            load_type:    String indicating whether to load training or validation data ('train' or 'val') 
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
        self.json_path      = kwargs['json_path']
        self.load_type      = kwargs['load_type']
        
        # Clips processing arguments
        self.clip_length    = kwargs['clip_length']
        self.clip_offset    = kwargs['clip_offset']
        self.clip_stride    = kwargs['clip_stride']
        self.num_clips      = kwargs['num_clips']

        # Frame-wise processing arguments
        self.resize_shape   = kwargs['resize_shape']
        self.crop_shape     = kwargs['crop_shape'] 
        self.crop_type      = kwargs['crop_type'] 
        self.final_shape    = kwargs['final_shape']

        # Creates the self.samples list which will be indexed by each __getitem__ call
        self._getClips()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raise NotImplementedError("Dataset must contain __getitem__ method which loads videos from memory.")

    def _getClips(self):
        """
        Loads the JSON file associated with the videos in a datasets and processes each of them into clips
        """
        raise NotImplementedError("Dataset must contain getClips method which loads and processes the dataset's JSON file.") 

    def _extractClips(self, video):
        """
        Processes a single video into uniform sized clips that will be loaded by __getitem__
        Args:
            video:       List containing a dictionary of annontations per frame

        Additional Parameters:
            self.clip_length: Number of frames extracted from each clip
            self.num_clips:   Number of clips to extract from each video (-1 uses the entire video, 0 paritions the entire video in clip_length clips)
            self.clip_offset: Number of frames from beginning of video to start extracting clips 
            self.clip_stride: Number of frames between clips when extracting them from videos 
            self.random_offset: Randomly select a clip_length sized clip from a video
        """
        if self.num_clips < 0:
            if len(video) >= self.clip_length:
                final_video = [video[_idx] for _idx in np.linspace(0, len(video)-1, self.clip_length, dtype='int32')]

            else:
                # Loop if insufficient elements
                indices = np.ceil(self.clip_length/float(len(video)))
                indices = indices.astype('int32')
                indices = np.tile(np.arange(0, len(video), 1, dtype='int32'), indices)
                indices = indices[np.linspace(0, len(indices)-1, self.clip_length, dtype='int32')]

                final_video = [video[_idx] for _idx in indices]


            # END IF

        elif self.num_clips == 0:
            if len(video) >= self.clip_length:
                indices     = np.arange(start=0, stop=len(video), step=self.clip_length)
                final_video = []

                for _idx in indices:
                    if _idx + self.clip_length <= len(video):
                        final_video.append([video[true_idx] for true_idx in range(_idx, _idx+self.clip_length)])

                # END FOR

            else:
                # Loop if insufficient elements
                indices = np.ceil(self.clip_length/float(len(video)))
                indices = indices.astype('int32')
                indices = np.tile(np.arange(0, len(video), 1, dtype='int32'), indices)
                indices = indices[:self.clip_length]

                final_video = [video[_idx] for _idx in indices]

            # END IF                               
    
        else:
            if self.random_offset:
                if len(video) >= self.clip_length:
                    indices = np.random.choice(np.arange(len(video) - self.clip_length), 1)
                    indices = indices.astype('int32')
                    indices = np.arange(indices, indices + self.clip_length).astype('int32') 

                    final_video = [video[_idx] for _idx in indices]

                else:
                    indices = np.ceil(self.clip_length/float(len(video)))
                    indices = indices.astype('int32')
                    indices = np.tile(np.arange(0, len(video), 1, dtype='int32'), indices)

                    index   = np.random.choice(np.arange(len(indices) - self.clip_length), 1)
                    index   = index.astype('int32')
                    indices = indices[index:index + self.clip_length]

                    final_video = [video[_idx] for _idx in indices]

                # END IF

            else:
                final_video = video[:self.clip_length]

            # END IF

        # END IF

        return [final_video]


        


class RecognitionDataset(VideoDataset):
    __metaclass__ = ABCMeta
    def __init__(self, *args, **kwargs):
        super(RecognitionDataset, self).__init__(*args, **kwargs)
        self.load_type = kwargs['load_type']

    def _getClips(self, *args, **kwargs):
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

        self.samples   = []
 
        if self.load_type == 'train':
            full_json_path = os.path.join(self.json_path, 'train.json')

        elif self.load_type == 'val':
            full_json_path = os.path.join(self.json_path, 'val.json') 

        else:
            full_json_path = os.path.join(self.json_path, 'test.json')

        # END IF 

        json_file = open(full_json_path,'r')
        json_data = json.load(json_file) 
        json_file.close()

        # Load the information for each video and process it into clips
        for video_info in json_data:
            clips = self._extractClips(video_info['frames'])

            # Each clip is a list of dictionaries per frame containing information
            # Example info: object bbox annotations, object classes, frame img path
            for clip in clips:    
                self.samples.append(dict(frames=clip, base_path=video_info['base_path']))



class DetectionDataset(VideoDataset):
    __metaclass__ = ABCMeta
    def __init__(self, *args, **kwargs):
        super(DetectionDataset, self).__init__(*args, **kwargs)
        self.load_type = kwargs['load_type']

    def _getClips(self):
        """
        Required format for all detection datset JSON files:
        Json (List of dicts) List where each element contains a dict with annotations for a video:
            Dict{
            'frame_size' (int,int): Width, Height for all frames in video
            'base_path' (str): The path to the folder containing frame images for the video
            'frame' (List of dicts): A list with annotation dicts per frame
                Dict{
                'img_path' (Str): File name of the image corresponding to the frame annotations
                'objs' (List of dicts): A list of dicts containing annotations for each object in the frame  
                    Dict{
                    'trackid' (Int): Id of the current object
                    'c' (Str or Int): Value indicating the class of the current object 
                    'bbox' (int, int, int, int): Bbox coordinates of the current object in the current frame (xmin, ymin, xmax, ymax)
                    (Optional) 'iscrowd' (int): Boolean indicating if the object represents a crowed (Used in MSCOCO dataset)
                    (Optional) 'occ' (int): Boolean indicating if the object is occluded in the current frame (Used in ImageNetVID dataset)
                    }
                }
            }
            
        
        Ex: coordinates = dataset[vid_index]['frame'][frame_index]['objs'][obj_index]['bbox']
        """

        # Load all video paths into the samples array to be loaded by __getitem__ 

        self.samples   = []
        
        if self.load_type == 'train':
            full_json_path = os.path.join(self.json_path, 'train.json')

        else:
            full_json_path = os.path.join(self.json_path, 'val.json') 

        json_file = open(full_json_path,'r')
        json_data = json.load(json_file) 
        json_file.close()

        # Load the information for each video and process it into clips
        for video_info in json_data:
            clips = self._extractClips(video_info['frame'])

            # Each clip is a list of dictionaries per frame containing information
            # Example info: object bbox annotations, object classes, frame img path
            for clip in clips:    
                self.samples.append(dict(frames=clip, base_path=video_info['base_path'], frame_size=video_info['frame_size']))


