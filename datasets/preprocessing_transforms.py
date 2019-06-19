"""
Functions used to process and augment video data prior to passing into a model to train. 
Additionally also processing all bounding boxes in a video according to the transformations performed on the video.

Usage:
    In a custom dataset class:
    from preprocessing_transforms import *

clip: Input to __call__ of each transform is a list of PIL images
"""

import torch
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
from abc import ABCMeta

class PreprocTransform(object):
    """
    Abstract class for preprocessing transforms that contains methods to convert clips to PIL images.
    """
    __metaclass__ = ABCMeta
    def __init__(self):
        self.image_type = type(Image.Image())
        self.numpy_type = type(np.array(0))

    def _format_clip(self, clip):
        assert((type(clip)==type(list())) or (type(clip)==self.numpy_type)), "Clips input to preprocessing transforms must be a list of PIL Images or numpy arrays"
        output_clip = []
        if type(clip[0]) == self.image_type:
            output_clip = clip

        else:
            for frame in clip:
                if len(frame.shape)==3:
                    output_clip.append(Image.fromarray(frame, mode='RGB'))
                else:
                    output_clip.append(Image.fromarray(frame))


        return output_clip
    
    def _format_clip_numpy(self, clip):
        assert(type(clip)==type(list())), "Clip must be a list when input to _format_clip_numpy"
        output_clip = []
        if type(clip[0]) == self.numpy_type:
            output_clip = clip

        else:
            for frame in clip:
                output_clip.append(np.array(frame))

        return np.array(output_clip)





class ResizeClip(PreprocTransform):
    def __init__(self, output_size, *args, **kwargs):
        super(ResizeClip, self).__init__(*args, **kwargs)

        self.size_h, self.size_w = output_size
        
    def __call__(self, clip, bbox=[]):
        clip = self._format_clip(clip)
        out_clip = []
        out_bbox = []
        for frame_ind in range(len(clip)):
            frame = clip[frame_ind]
            if type(frame) != self.image_type:
                frame = Image.fromarray(frame)

            proc_frame = frame.resize((self.size_w, self.size_h))
            out_clip.append(proc_frame)
            if bbox!=[]:
                xmin, ymin, xmax, ymax = bbox[frame_ind]
                proc_bbox = resize_bbox(xmin, ymin, xmax, ymax, frame.size, (self.size_w, self.size_h))
                out_bbox.append(proc_bbox)

        if bbox!=[]:
            return out_clip, out_bbox
        else:
            return out_clip


class CropClip(PreprocTransform):
    def __init__(self, xmin, xmax, ymin, ymax, *args, **kwargs):
        super(CropClip, self).__init__(*args, **kwargs)
        self.bbox_xmin = xmin
        self.bbox_xmax = xmax
        self.bbox_ymin = ymin
        self.bbox_ymax = ymax

        self.image_type = type(Image.Image())

    def _update_bbox(self, xmin, xmax, ymin, ymax):
        self.bbox_xmin = xmin
        self.bbox_xmax = xmax
        self.bbox_ymin = ymin
        self.bbox_ymax = ymax

        
    def __call__(self, clip, bbox=[]):
        clip = self._format_clip(clip)
        out_clip = []
        out_bbox = []

        for frame_ind in range(len(clip)):
            frame = clip[frame_ind]
            proc_frame = frame.crop((self.bbox_xmin, self.bbox_ymin, self.bbox_xmax, self.bbox_ymax))
            out_clip.append(proc_frame)

            if bbox!=[]:
                xmin, ymin, xmax, ymax = bbox[frame_ind]
                proc_bbox = crop_bbox(xmin, ymin, xmax, ymax, self.bbox_xmin, self.bbox_xmax, self.bbox_ymin, self.bbox_ymax)
                out_bbox.append(proc_bbox)

        if bbox!=[]:
            return out_clip, out_bbox
        else:
            return out_clip


class RandomCropClip(PreprocTransform):
    def __init__(self, crop_w, crop_h, *args, **kwargs):
        super(RandomCropClip, self).__init__(*args, **kwargs)
        self.crop_w = crop_w 
        self.crop_h = crop_h

        self.crop_transform = CropClip(0, 0, self.crop_w, self.crop_h)

        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None


    def _update_random_sample(self, frame_h, frame_w):
        self.xmin = np.random.randint(0, frame_w-self.crop_w)
        self.xmax = self.xmin + self.crop_w
        self.ymin = np.random.randint(0, frame_h-self.crop_h)
        self.ymax = self.ymin + self.crop_h

    def get_random_sample(self):
        return self.xmin, self.xmax, self.ymin, self.ymax
        
    def __call__(self, clip, bbox=[]):
        clip = self._format_clip(clip)
        frame_shape = clip[0].size
        self._update_random_sample(frame_shape[0], frame_shape[1])
        self.crop_transform._update_bbox(self.xmin, self.xmax, self.ymin, self.ymax) 
        return self.crop_transform(clip, bbox)


class CenterCropClip(PreprocTransform):
    def __init__(self, crop_w, crop_h, *args, **kwargs):
        super(CenterCropClip, self).__init__(*args, **kwargs)
        self.crop_w = crop_w 
        self.crop_h = crop_h

        self.crop_transform = CropClip(0, 0, self.crop_w, self.crop_h)

    def _calculate_center(self, frame_h, frame_w):
        xmin = int(frame_w/2 - self.crop_w/2)
        xmax = int(frame_w/2 + self.crop_w/2)
        ymin = int(frame_h/2 - self.crop_h/2)
        ymax = int(frame_h/2 + self.crop_h/2)
        return xmin, xmax, ymin, ymax
        
    def __call__(self, clip, bbox=[]):
        clip = self._format_clip(clip)
        frame_shape = clip[0].size
        xmin, xmax, ymin, ymax = self._calculate_center(frame_shape[0], frame_shape[1])
        self.crop_transform._update_bbox(xmin, xmax, ymin, ymax) 
        return self.crop_transform(clip, bbox)


class RandomFlipClip(PreprocTransform):
    """
    Specify a flip direction:
    Horizontal, left right flip = 'h' (Default)
    Vertical, top bottom flip = 'v'
    """
    def __init__(self, direction='h', p=0.5, *args, **kwargs):
        super(RandomFlipClip, self).__init__(*args, **kwargs)
        self.direction = direction
        self.p = p
            
    def _random_flip(self):
        flip_prob = np.random.random()
        if flip_prob >= self.p:
            return 0
        else:
            return 1

    def _h_flip(self, bbox, frame_size):
        xmin, ymin, xmax, ymax = bbox 
        width = frame_size[1]
        xmax_new = width - xmin 
        xmin_new = width - xmax
        return [xmin_new, ymin, xmax_new, ymax]

    def _v_flip(self, bbox, frame_size):
        xmin, ymin, xmax, ymax = bbox 
        height = frame_size[0]
        ymax_new = height - ymin 
        ymin_new = height - ymax
        return [xmin, ymin_new, xmax, ymax_new]


    def _flip_data(self, clip, bbox=[]):
        output_bbox = []
        
        if self.direction == 'h':
            output_clip = [frame.transpose(Image.FLIP_LEFT_RIGHT) for frame in clip]
            
            if bbox!=[]:
                output_bbox = [self._h_flip(curr_bbox, output_clip[0].size) for curr_bbox in bbox] 

        elif self.direction == 'v':
            output_clip = [frame.transpose(Image.FLIP_TOP_BOTTOM) for frame in clip]

            if bbox!=[]:
                output_bbox = [self._v_flip(curr_bbox, output_clip[0].size) for curr_bbox in bbox]

        return output_clip, output_bbox 
        

    def __call__(self, clip, bbox=[]):
        clip = self._format_clip(clip)
        flip = self._random_flip()
        out_clip = clip
        out_bbox = bbox
        if flip:
            out_clip, out_bbox = self._flip_data(clip, bbox)

        if bbox!=[]:
            return out_clip, out_bbox
        else:
            return out_clip

class ToTensorClip(PreprocTransform):
    """
    Convert a list of PIL images or numpy arrays to a 5 dimensional pytorch tensor [batch, frame, channel, height, width]
    """
    def __init__(self, *args, **kwargs):
        super(ToTensorClip, self).__init__(*args, **kwargs)

    def __call__(self, clip, bbox=[]):
        clip = self._format_clip_numpy(clip)
        clip = torch.from_numpy(clip)
        if bbox!=[]:
            bbox = torch.from_numpy(np.array(bbox))
            return clip, bbox
        else:
            return clip
        


#class oversample(object):
#    def __init__(self, output_size):
#        self.size_h, self.size_w = output_size
#        
#    def __call__(self, clip, bbox):
#        return clip, bbox


#class subtractMean(object):
#    def __init__(self, output_size):
#        self.size_h, self.size_w = output_size
#        
#    def __call__(self, clip, bbox):
#        return clip, bbox


def resize_bbox(xmin, xmax, ymin, ymax, img_shape, resize_shape):
    # Resize a bounding box within a frame relative to the amount that the frame was resized

    img_h = img_shape[0]
    img_w = img_shape[1]

    res_h = resize_shape[0]
    res_w = resize_shape[1]

    frac_h = res_h/float(img_h)
    frac_w = res_w/float(img_w)

    xmin_new = int(xmin * frac_w)
    xmax_new = int(xmax * frac_w)

    ymin_new = int(ymin * frac_h)
    ymax_new = int(ymax * frac_h)

    return xmin_new, xmax_new, ymin_new, ymax_new 


def crop_bbox(xmin, xmax, ymin, ymax, crop_xmin, crop_xmax, crop_ymin, crop_ymax):
    if (xmin >= crop_xmax) or (xmax <= crop_xmin) or (ymin >= crop_ymax) or (ymax <= crop_ymin):
        return -1, -1, -1, -1

    if ymax > crop_ymax:
        ymax_new = crop_ymax
    else:
        ymax_new = ymax

    if xmax > crop_xmax:
        xmax_new = crop_xmax
    else:
        xmax_new = xmax
    
    if ymin < crop_ymin:
        ymin_new = crop_ymin
    else:
        ymin_new = ymin 

    if xmin < crop_xmin:
        xmin_new = crop_xmin
    else:
        xmin_new = xmin 

    return xmin_new, xmax_new, ymin_new, ymax_new


