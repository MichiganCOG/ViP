"""
Functions used to process and augment video data prior to passing into a model to train. 
Additionally also processing all bounding boxes in a video according to the transformations performed on the video.

Usage:
    In a custom dataset class:
    from preprocessing_transforms import *

clip: Input to __call__ of each transform is a list of PIL images
"""

from torchvision.transforms import functional as F
from PIL import Image

class resizeClip(object):
    def __init__(self, output_size):
        self.size_h, self.size_w = output_size
        
    def __call__(self, clip, bbox):
        out_clip = []
        out_bbox = []
        for frame_ind in range(len(clip)):
            frame = clip[frame_ind]
            xmin, ymin, xmax, ymax = bbox[frame_ind]
            proc_frame = frame.resize((self.size_w, self.size_h), Image.ANTIALiAS)
            out_clip.append(proc_frame)
            proc_bbox = resize_bbox(xmin, ymin, xmax, ymax, frame.shape, (self.size_w, self.size_h))
            out_bbox.append(proc_bbox)

        return out_clip, out_bbox


class cropClip(object):
    def __init__(self, xmin, xmax, ymin, ymax):
        self.bbox_xmin = xmin
        self.bbox_xmax = xmax
        self.bbox_ymin = ymin
        self.bbox_ymax = ymax

    def update_bbox(self, xmin, xmax, ymin, ymax):
        self.bbox_xmin = xmin
        self.bbox_xmax = xmax
        self.bbox_ymin = ymin
        self.bbox_ymax = ymax

        
    def __call__(self, clip, bbox):
        out_clip = []
        out_bbox = []

        for frame_ind in range(len(clip)):
            frame = clip[frame_ind]
            xmin, ymin, xmax, ymax = bbox[frame_ind]
            proc_frame = frame.crop((self.bbox_xmin, self.bbox_ymin, self.bbox_xmax, self.bbox_ymax))
            out_clip.append(proc_frame)
            proc_bbox = crop_bbox(xmin, ymin, xmax, ymax, self.bbox_xmin, self.bbox_xmax, self.bbox_ymin, self.bbox_ymax)
            out_bbox.append(proc_bbox)

        return out_clip, out_bbox


class randomCropClip(object):
    def __init__(self, crop_w, crop_h):
        self.crop_w = crop_w 
        self.crop_h = crop_h

        self.crop_transform = cropClip(0, 0, self.crop_w, self.crop_h)

    def random_sample(self, frame_h, frame_w):
        xmin = np.random.randint(0, frame_w-self.crop_w)
        xmax = xmin + self.crop_w
        ymin = np.random.randint(0, frame_h-self.crop_h)
        ymax = ymin + self.crop_h
        return xmin, xmax, ymin, ymax
        
    def __call__(self, clip, bbox):
        frame_shape = clip[0].shape
        xmin, xmax, ymin, ymax = self.random_sample(frame_shape[0], frame_shape[1])
        self.crop_transform.update_bbox(self, xmin, xmax, ymin, ymax) 
        return self.crop_transform(clip, bbox)


class centerCropClip(object):
    def __init__(self, crop_w, crop_h):
        self.crop_w = crop_w 
        self.crop_h = crop_h

        self.crop_transform = cropClip(0, 0, self.crop_w, self.crop_h)

    def calculate_center(self, frame_h, frame_w):
        xmin = int(frame_w/2 - self.crop_w/2)
        xmax = int(frame_w/2 + self.crop_w/2)
        ymin = int(frame_h/2 - self.crop_h/2)
        ymax = int(frame_h/2 + self.crop_h/2)
        return xmin, xmax, ymin, ymax
        
    def __call__(self, clip, bbox):
        frame_shape = clip[0].shape
        xmin, xmax, ymin, ymax = self.calculate_center(frame_shape[0], frame_shape[1])
        self.crop_transform.update_bbox(self, xmin, xmax, ymin, ymax) 
        return self.crop_transform(clip, bbox)


class randomFlipClip(object):
    """
    Specify a flip direction:
    Horizontal, left right flip = 'h' (Default)
    Vertical, top bottom flip = 'v'
    """
    def __init__(self, direction='h', p=0.5):
        self.direction = direction
        self.p = p
            
    def random_flip(self):
        flip_prob = np.random.random()
        if flip_prob >= self.p:
            return 0
        else:
            return 1

    def h_flip(self, bbox, frame_size):
        xmin, ymin, xmax, ymax = bbox 
        width = frame_size[1]
        xmax_new = width - xmin 
        xmin_new = width - xmax
        return [xmin_new, ymin, xmax_new, ymax]

    def v_flip(self, bbox, frame_size):
        xmin, ymin, xmax, ymax = bbox 
        height = frame_size[0]
        ymax_new = height - ymin 
        ymin_new = height - ymax
        return [xmin, ymin_new, xmax, ymax_new]


    def flip_data(self, clip, bbox):
        
        if self.direction == 'h':
            output_clip = [frame.transpose(PIL.Image.FLIP_LEFT_RIGHT) for frame in clip]
            output_bbox = [h_flip(curr_bbox, output_clip[0].shape) for curr_bbox in bbox] 

        elif self.direction == 'v':
            output_clip = [frame.transpose(PIL.Image.FLIP_TOP_BOTTOM) for frame in clip]
            output_bbox = [v_flip(curr_bbox, output_clip[0].shape) for curr_bbox in bbox]

        return output_clip, output_bbox 
        

    def __call__(self, clip, bbox):
        flip = random_flip()
        output_clip = clip
        output_bbox = bbox
        if flip:
            output_clip, output_bbox = flip_data(clip, bbox)

        return clip, bbox


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


def crop_bbox(self, xmin, xmax, ymin, ymax, crop_xmin, crop_xmax, crop_ymin, crop_ymax):
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


