"""
Functions used to process and augment video data prior to passing into a model to train. 
Additionally also processing all bounding boxes in a video according to the transformations performed on the video.

Usage:
    In a custom dataset class:
    from preprocessing_functions import *
"""

import torchvision

def resizeClip(clip, bbox):
    return clip, bbox

def cropClip(clip, bbox):
    return clip, bbox

def randomCropClip(clip, bbox):
    return clip, bbox

def centerCropClip(clip, bbox):
    return clip, bbox

def hRandomFlipClip(clip, bbox):
    return clip, bbox

def oversample(clip, bbox):
    return clip, bbox

def subtractMean(clip, bbox):
    return clip, bbox

def resize_bbox(self, xmin, xmax, ymin, ymax, img_shape, resize_shape):
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
