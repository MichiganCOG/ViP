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
from PIL import ImageChops
import cv2
from scipy import ndimage
import numpy as np
from abc import ABCMeta
from math import floor, ceil

class PreprocTransform(object):
    """
    Abstract class for preprocessing transforms that contains methods to convert clips to PIL images.
    """
    __metaclass__ = ABCMeta
    def __init__(self, **kwargs):
        pass

    def __init__(self, *args, **kwargs):
        pass

    def _to_pil(self, clip):
        output=[]
        for frame in clip:
            output.append(F._to_pil_image(frame))
        
        return output


    def _to_numpy(self, clip):
        output = []
        if isinstance(clip[0], torch.Tensor):
            if isinstance(clip, torch.Tensor):
                output = clip.numpy()
            else:
                for frame in clip:
                    output.append(frame.numpy())
            

        elif isinstance(clip[0], Image.Image):
            for frame in clip:
                output.append(np.array(frame))

        else:
            output = clip 

        output = np.array(output)

        if output.max() > 1.0:
            output = output/255.

        return output


    def _to_tensor(self, clip):
            

        output = []
        for frame in clip:
            output.append(F.to_tensor(frame))
        
        return output

    def _format_clip(self, clip):

        assert((type(clip)==type(list())) or (type(clip)==self.numpy_type)), "Clips input to preprocessing transforms must be a list of PIL Images or numpy arrays"
        output_clip = []
        
        if type(clip[0]) == self.numpy_type:
            for frame in clip:
                if len(frame.size)==3:
                    output_clip.append(Image.fromarray(frame, mode='RGB'))
                else:
                    import pdb; pdb.set_trace()
                    output_clip.append(Image.fromarray(frame))
        else:
            output_clip = clip


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
    def __init__(self, *args, **kwargs):
        super(ResizeClip, self).__init__(*args, **kwargs)
        self.size_h, self.size_w = kwargs['resize_shape']
        
    def __call__(self, clip, bbox=[]):

        #clip = self._format_clip(clip)
        clip = self._to_numpy(clip)
        out_clip = []
        out_bbox = []
        for frame_ind in range(len(clip)):
            frame = clip[frame_ind]

            proc_frame = cv2.resize(frame, (self.size_w, self.size_h))
            out_clip.append(proc_frame)
            if bbox!=[]:
                temp_bbox = np.zeros(bbox[frame_ind].shape)-1 
                for class_ind in range(len(bbox[frame_ind])):
                    if np.array_equal(bbox[frame_ind,class_ind],-1*np.ones(4)): #only annotated objects
                        continue
                    xmin, ymin, xmax, ymax = bbox[frame_ind, class_ind]
                    proc_bbox = resize_bbox(xmin, ymin, xmax, ymax, frame.shape, (self.size_w, self.size_h))
                    temp_bbox[class_ind,:] = proc_bbox
                out_bbox.append(temp_bbox)

        if bbox!=[]:
            return np.array(out_clip), np.array(out_bbox)
        else:
            return np.array(out_clip)


class CropClip(PreprocTransform):
    def __init__(self, xmin, xmax, ymin, ymax, *args, **kwargs):
        super(CropClip, self).__init__(*args, **kwargs)
        self.bbox_xmin = xmin
        self.bbox_xmax = xmax
        self.bbox_ymin = ymin
        self.bbox_ymax = ymax


    def _update_bbox(self, xmin, xmax, ymin, ymax):
        self.bbox_xmin = xmin
        self.bbox_xmax = xmax
        self.bbox_ymin = ymin
        self.bbox_ymax = ymax

        
    def __call__(self, clip, bbox=[]):
        #clip = self._format_clip(clip)
        out_clip = []
        out_bbox = []

        for frame_ind in range(len(clip)):
            frame = clip[frame_ind]
            proc_frame = np.array(frame[self.bbox_xmin:self.bbox_xmax, self.bbox_ymin:self.bbox_ymax])   #frame.crop((self.bbox_xmin, self.bbox_ymin, self.bbox_xmax, self.bbox_ymax))
            out_clip.append(proc_frame)

            if bbox!=[]:
                temp_bbox = np.zeros(bbox[frame_ind].shape)-1 
                for class_ind in range(len(bbox)):
                    if np.array_equal(bbox[frame_ind,class_ind],-1*np.ones(4)): #only annotated objects
                        continue
                    xmin, ymin, xmax, ymax = bbox[frame_ind, class_ind]
                    proc_bbox = crop_bbox(xmin, ymin, xmax, ymax, self.bbox_xmin, self.bbox_xmax, self.bbox_ymin, self.bbox_ymax)
                    temp_bbox[class_ind,:] = proc_bbox
                out_bbox.append(temp_bbox)

        if bbox!=[]:
            return np.array(out_clip), np.array(out_bbox)
        else:
            return np.array(out_clip)


class RandomCropClip(PreprocTransform):
    def __init__(self, *  args, **kwargs):
        super(RandomCropClip, self).__init__(*args, **kwargs)
        self.crop_h, self.crop_w = kwargs['crop_shape']

        self.crop_transform = CropClip(0, 0, self.crop_w, self.crop_h)

        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None


    def _update_random_sample(self, frame_w, frame_h):
        self.xmin = np.random.randint(0, frame_w-self.crop_w)
        self.xmax = self.xmin + self.crop_w
        self.ymin = np.random.randint(0, frame_h-self.crop_h)
        self.ymax = self.ymin + self.crop_h

    def get_random_sample(self):
        return self.xmin, self.xmax, self.ymin, self.ymax
        
    def __call__(self, clip, bbox=[]):
        #clip = self._format_clip(clip)
        frame_shape = clip[0].shape
        self._update_random_sample(frame_shape[0], frame_shape[1])
        self.crop_transform._update_bbox(self.xmin, self.xmax, self.ymin, self.ymax) 
        return self.crop_transform(clip, bbox)


class CenterCropClip(PreprocTransform):
    def __init__(self, *args, **kwargs):
        super(CenterCropClip, self).__init__(*args, **kwargs)
        self.crop_h, self.crop_w = kwargs['crop_shape']

        self.crop_transform = CropClip(0, 0, self.crop_w, self.crop_h)

    def _calculate_center(self, frame_w, frame_h):
        xmin = int(frame_w/2 - self.crop_w/2)
        xmax = int(frame_w/2 + self.crop_w/2)
        ymin = int(frame_h/2 - self.crop_h/2)
        ymax = int(frame_h/2 + self.crop_h/2)
        return xmin, xmax, ymin, ymax
        
    def __call__(self, clip, bbox=[]):
        #clip = self._format_clip(clip)
        frame_shape = clip[0].shape
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
        bbox_shape = bbox.shape
        output_bbox = np.zeros(bbox_shape)-1
        for bbox_ind in range(bbox_shape[0]):
            xmin, ymin, xmax, ymax = bbox[bbox_ind] 
            width = frame_size[1]
            xmax_new = width - xmin 
            xmin_new = width - xmax
            output_bbox[bbox_ind] = xmin_new, ymin, xmax_new, ymax
        return output_bbox 

    def _v_flip(self, bbox, frame_size):
        bbox_shape = bbox.shape
        output_bbox = np.zeros(bbox_shape)-1
        for bbox_ind in range(bbox_shape[0]):
            xmin, ymin, xmax, ymax = bbox[bbox_ind] 
            height = frame_size[0]
            ymax_new = height - ymin 
            ymin_new = height - ymax
            output_bbox[bbox_ind] = xmin_new, ymin, xmax_new, ymax
        return output_bbox 


    def _flip_data(self, clip, bbox=[]):
        output_bbox = []
        
        if self.direction == 'h':
            #output_clip = [frame.transpose(Image.FLIP_LEFT_RIGHT) for frame in clip]
            output_clip = [cv2.flip(np.array(frame), 1) for frame in clip]
            
            if bbox!=[]:
                output_bbox = [self._h_flip(curr_bbox, output_clip[0].size) for curr_bbox in bbox] 

        elif self.direction == 'v':
            #output_clip = [frame.transpose(Image.FLIP_TOP_BOTTOM) for frame in clip]
            output_clip = [cv2.flip(np.array(frame), 0) for frame in clip]

            if bbox!=[]:
                output_bbox = [self._v_flip(curr_bbox, output_clip[0].size) for curr_bbox in bbox]

        return output_clip, output_bbox 
        

    def __call__(self, clip, bbox=[]):
        #clip = self._format_clip(clip)
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
        #clip = self._format_clip_numpy(clip)
        clip = torch.from_numpy(np.array(clip)).float()
        if bbox!=[]:
            bbox = torch.from_numpy(np.array(bbox))
            return clip, bbox
        else:
            return clip
        

class RandomRotateClip(PreprocTransform):
    """
    Randomly rotate a clip from a fixed set of angles.
    The rotation is counterclockwise
    """
    def __init__(self,  angles=[0,90,180,270], *args, **kwargs):
        super(RandomRotateClip, self).__init__(*args, **kwargs)
        self.angles = angles

    ######
    # Code from: https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
    def _cart2pol(self, point):
        x,y = point
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)
    
    def _pol2cart(self, point):
        rho, phi = point
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return(x, y)
    #####

    def _update_angles(self, angles):
        self.angles=angles


    def _rotate_bbox(self, bboxes, frame_shape, angle):
        angle = np.deg2rad(angle)
        bboxes_shape = bboxes.shape
        output_bboxes = np.zeros(bboxes_shape)-1
        frame_h, frame_w = frame_shape 
        half_h = frame_h/2. 
        half_w = frame_w/2. 

        for bbox_ind in range(bboxes_shape[0]):
            xmin, ymin, xmax, ymax = bboxes[bbox_ind]
            tl = (xmin-half_w, ymax-half_h)
            tr = (xmax-half_w, ymax-half_h)
            bl = (xmin-half_w, ymin-half_h)
            br = (xmax-half_w, ymin-half_h)

            tl = self._cart2pol(tl) 
            tr = self._cart2pol(tr)    
            bl = self._cart2pol(bl)
            br = self._cart2pol(br)

            tl = (tl[0], tl[1] - angle)
            tr = (tr[0], tr[1] - angle)
            bl = (bl[0], bl[1] - angle)
            br = (br[0], br[1] - angle)

            tl = self._pol2cart(tl) 
            tr = self._pol2cart(tr)    
            bl = self._pol2cart(bl)
            br = self._pol2cart(br)

            tl = (tl[0]+half_w, tl[1]+half_h)
            tr = (tr[0]+half_w, tr[1]+half_h)
            bl = (bl[0]+half_w, bl[1]+half_h)
            br = (br[0]+half_w, br[1]+half_h)

            xmin_new = int(min(floor(tl[0]), floor(tr[0]), floor(bl[0]), floor(br[0])))
            xmax_new = int(max(ceil(tl[0]), ceil(tr[0]), ceil(bl[0]), ceil(br[0])))
            ymin_new = int(min(floor(tl[1]), floor(tr[1]), floor(bl[1]), floor(br[1])))
            ymax_new = int(max(ceil(tl[1]), ceil(tr[1]), ceil(bl[1]), ceil(br[1])))

            output_bboxes[bbox_ind] = [xmin_new, ymin_new, xmax_new, ymax_new]

        return output_bboxes



    def __call__(self, clip, bbox=[]):
        angle = np.random.choice(self.angles)
        output_clip = []
        clip = self._to_numpy(clip)
        for frame in clip:
            #output_clip.append(frame.rotate(angle))
            output_clip.append(ndimage.rotate(frame, angle, reshape=False))

        if bbox!=[]:
            bbox = np.array(bbox)
            output_bboxes = np.zeros(bbox.shape)-1
            output_bboxes = self._rotate_bbox(bbox, clip[0].shape, angle)

            return output_clip, output_bboxes 

        return output_clip



#class oversample(object):
#    def __init__(self, output_size):
#        self.size_h, self.size_w = output_size
#        
#    def __call__(self, clip, bbox):
#        return clip, bbox


class SubtractMeanClip(PreprocTransform):
    def __init__(self, **kwargs):
        super(SubtractMeanClip, self).__init__(**kwargs)
        self.clip_mean = torch.tensor(kwargs['clip_mean']).float()
#        self.clip_mean_args = kwargs['clip_mean']
#        self.clip_mean      = []
#
#        for frame in self.clip_mean_args:
#            self.clip_mean.append(Image.fromarray(frame))

        
    def __call__(self, clip, bbox=[]):
        clip = clip-self.clip_mean
        #for clip_ind in range(len(clip)):
        #    clip[clip_ind] = ImageChops.subtract(clip[clip_ind], self.clip_mean[clip_ind])

        
        if bbox!=[]:
            return clip, bbox

        else:
            return clip


class ApplyToPIL(PreprocTransform):
    """
    Apply standard pytorch transforms that require PIL images as input to their __call__ function, for example Resize

    NOTE: The __call__ function of this class converts the clip to a list of PIL images in the form of integers from 0-255. If the clips are floats (for example afer mean subtraction), then only call this transform before the float transform

    Bounding box coordinates are not guaranteed to be transformed properly!

    https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html
    """
    def __init__(self, **kwargs):
        super(ApplyToPIL, self).__init__(**kwargs)
        self.kwargs = kwargs
        self.transform = kwargs['transform']

    def __call__(self, clip, bbox=[]):
        if not isinstance(clip[0], Image.Image):
            clip = self._to_pil(clip)
            if self.kwargs['verbose']:
                print("Clip has been converted to PIL from numpy or tensor.")
        output_clip = []
        for frame in clip:
            output_clip.append(self.transform(frame))

        if bbox!=[]:
            return output_clip, bbox

        else:
            return output_clip


class ApplyToTensor(PreprocTransform):
    """
    Apply standard pytorch transforms that require pytorch Tensors as input to their __call__ function, for example Normalize 

    NOTE: The __call__ function of this class converts the clip to a pytorch float tensor. If other transforms require PIL inputs, call them prior tho this one
    Bounding box coordinates are not guaranteed to be transformed properly!

    https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html
    """
    def __init__(self, **kwargs):
        super(ApplyToTensor, self).__init__(**kwargs)
        self.kwargs = kwargs
        self.transform = kwargs['transform']

    def __call__(self, clip, bbox=[]):
        if not isinstance(clip, torch.Tensor):
            clip = self._to_tensor(clip)
            if self.kwargs['verbose']:
                print("Clip has been converted to tensor from numpy or PIL.")
        output_clip = []
        for frame in clip:
            output_clip.append(self.transform(frame))

        output_clip = torch.stack(output_clip)

        if bbox!=[]:
            return output_clip, bbox

        else:
            return output_clip

class ApplyOpenCV(PreprocTransform):
    """
    Apply opencv transforms that require numpy arrays as input to their __call__ function, for example Rotate 

    NOTE: The __call__ function of this class converts the clip to a Numpy array. If other transforms require PIL inputs, call them prior tho this one

    Bounding box coordinates are not guaranteed to be transformed properly!
    """
    def __init__(self, **kwargs):
        super(ApplyOpenCV, self).__init__(**kwargs)
        self.kwargs = kwargs
        self.function = kwargs['function']

    def __call__(self, clip, bbox=[]):
        if not isinstance(clip, torch.Tensor):
            clip = self._to_array(clip)
            if self.kwargs['verbose']:
                print("Clip has been converted to numpy from pytorch tensor or PIL.")
        output_clip = []
        for frame in clip:
            output_clip.append(self.function(frame))

        output_clip = torch.stack(output_clip)

        if bbox!=[]:
            return output_clip, bbox

        else:
            return output_clip

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


class TestPreproc(object):
    def __init__(self):
        self.resize = ResizeClip(resize_shape = [2,2])
        self.crop = CropClip(0,0,0,0)
        self.rand_crop = RandomCropClip(crop_shape=[2,2])
        self.cent_crop = CenterCropClip(crop_shape=[2,2])
        self.rand_flip_h = RandomFlipClip(direction='h', p=1.0)
        self.rand_flip_v = RandomFlipClip(direction='v', p=1.0)
        self.rand_rot = RandomRotateClip(angles=[90])
        self.sub_mean = SubtractMeanClip(clip_mean=np.zeros(1))

    def resize_test(self):
        inp = np.array([[[.1,.2,.3,.4],[.1,.2,.3,.4],[.1,.2,.3,.4]]]).astype(float)
        inp2 = np.array([[[.1,.1,.1,.1],[.2,.2,.2,.2],[.3,.3,.3,.3]]]).astype(float)
        expected_out = np.array([[[.15,.35],[.15,.35]]]).astype(float)
        expected_out2 = np.array([[[.125,.125],[.275,.275]]]).astype(float)
        out = self.resize(inp)
        out2 = self.resize(inp2)
        assert (False not in np.isclose(out,expected_out)) and (False not in np.isclose(out2,expected_out2))

    def crop_test(self):
        inp = np.array([[[.1,.2,.3],[.4,.5,.6],[.7,.8,.9]]]).astype(float)
        self.crop._update_bbox(1, 3, 1, 3)

        exp_out = np.array([[[.5,.6],[.8,.9]]]).astype(float)
        out = self.crop(inp)
        assert (False not in np.isclose(out,exp_out))

    def cent_crop_test(self):
        inp = np.array([[[.1,.2,.3,.4],[.1,.2,.3,.4],[.1,.2,.3,.4],[.1,.2,.3,.4]]]).astype(float)
        exp_out = np.array([[[.2,.3],[.2,.3]]]).astype(float)
        out = self.cent_crop(inp)
        assert (False not in np.isclose(out, exp_out))

    def rand_flip_test(self):
        inp = np.array([[[.1,.2,.3],[.4,.5,.6],[.7,.8,.9]]]).astype(float)
        exp_outh = np.array([[[.3,.2,.1],[.6,.5,.4],[.9,.8,.7]]]).astype(float)
        exp_outv = np.array([[[.7,.8,.9],[.4,.5,.6],[.1,.2,.3]]]).astype(float)

        outh = self.rand_flip_h(inp)
        outv = self.rand_flip_v(inp)
        assert (False not in np.isclose(outh,exp_outh)) and (False not in np.isclose(outv,exp_outv))

    def rand_rot_test(self):
        inp = np.array([[[.1,.2,.3],[.4,.5,.6],[.7,.8,.9]]]).astype(float)
        exp_out = np.array([[[.3,.6,.9],[.2,.5,.8],[.1,.4,.7]]]).astype(float)

        out = self.rand_rot(inp)
        assert (False not in np.isclose(out, exp_out))

    def rand_rot_vis(self):
        import matplotlib.pyplot as plt
        self.rand_rot._update_angles([20])
        x = np.arange(112*112).reshape(112,112)
        #x = np.arange(6*6).reshape(6,6)
        #bbox = [51,51,61,61]
        bbox = [30,40,50,100]
        #bbox = [2,2,4,4]
        plt1 = x[:]
        plt1[bbox[1]:bbox[3], bbox[0]] = 0
        plt1[bbox[1]:bbox[3], bbox[2]-1] = 0
        plt1[bbox[1], bbox[0]:bbox[2]] = 0
        plt1[bbox[3]-1, bbox[0]:bbox[2]] = 0
        plt.imshow(plt1); plt.show()
        out2 = self.rand_rot([x], np.array([bbox]))
        plt2 = out2[0][0]
        bbox = out2[1][0].astype(int)
        plt2[bbox[1]:bbox[3], bbox[0]] = 0
        plt2[bbox[1]:bbox[3], bbox[2]] = 0
        plt2[bbox[1], bbox[0]:bbox[2]] = 0
        plt2[bbox[3], bbox[0]:bbox[2]] = 0
        plt.imshow(plt2); plt.show()
        import pdb; pdb.set_trace()

    def run_tests(self):
        self.resize_test()
        self.crop_test()
        self.cent_crop_test()
        self.rand_flip_test()
        self.rand_rot_test()
        print("Tests passed")
        



if __name__=='__main__':
    test = TestPreproc()
    test.run_tests()


