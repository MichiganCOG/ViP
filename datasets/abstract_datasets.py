from abc import ABCMeta
from torch import Dataset

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

        self.samples = self.getClips()

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError("Dataset must contain __getitem__ method which loads videos from memory.")

    @abstractmethod
    def getClips(self):
        raise NotImplementedError("Dataset must contain getClips method which loads and processes the dataset's JSON file.") 
        


class RecognitionDataset(VideoDataset):
    __metaclass__ = ABCMeta
    def __init__(self, *args, **kwargs):
        super(RecognitionDataset, self).__init__(*args, **kwargs)

    def getClips(self):
        # TODO Implement JSON reader
        return []




class DetectionDataset(VideoDataset):
    __metaclass__ = ABCMeta
    def __init__(self, *args, **kwargs):
        super(RecognitionDataset, self).__init__(*args, **kwargs)

    def getClips(self):
        # TODO Implement JSON reader
        return []
