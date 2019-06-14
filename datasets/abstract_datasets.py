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

    def _extractClips(self, video):
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
        if self.num_clips < 0:
            if len(video) >= self.clip_length:
                final_video = [video[_idx] for _idx in np.linspace(0, len(video)-1, self.clip_length, dtype='int32')]
                return [final_video]

            else:
                # Loop if insufficient elements
                indices = np.ceil(self.clip_length/float(len(video)))
                indices = indices.astype('int32')
                indices = np.tile(np.arange(0, len(video), 1, dtype='int32'), indices)
                indices = indices[np.linspace(0, len(indices)-1, self.clip_length, dtype='int32')]
                final_video = [video[_idx] for _idx in indices]
                return final_video 

            # END IF

        else:
            return [video[:self.clip_length]]

        # END IF

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

        self.samples      = []
        self.dataset_type = 'train'
        
        if self.dataset_type == 'train':
            full_json_path = os.path.join(self.json_path, 'train.json')

        elif self.dataset_type == 'val':
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

        import pdb; pdb.set_trace()



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
            clips = self._extractClips(video_info['frame'])
            import pdb; pdb.set_trace()

            # Each clip is a list of dictionaries per frame containing information
            # Example info: object bbox annotations, object classes, frame img path
            for clip in clips:    
                self.samples.append(dict(frames=clip, base_path=video_info['base_path'], frame_size=video_info['frame_size']))


