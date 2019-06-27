import torch 
import torch.nn    as nn
import numpy as np

def play(vid):
    # Play a video stored as a 4d bgr numpy array (frame, h, w, channel)
    import matplotlib.pyplot as plt
    [(plt.imshow(frame[...,::-1]), plt.pause(0.0000001)) for frame in np.array(vid).astype(int)]

def plotbb(vid, xmin_in, xmax_in, ymin_in, ymax_in):
    # Play a video and plot a bounding box on it
    import matplotlib.pyplot as plt
    for frame_ind in range(len(np.array(vid).astype(int))):
        frame = np.array(vid).astype(int)[frame_ind]
        xmin = xmin_in[frame_ind,0]
        xmax = xmax_in[frame_ind,0]
        ymin = ymin_in[frame_ind,0]
        ymax = ymax_in[frame_ind,0]
        frame[ymin:ymax, xmin, 0] = 0
        frame[ymin:ymax, xmin, 1] = 0
        frame[ymin:ymax, xmin, 2] = 1
        frame[ymin:ymax, xmax, 0] = 0 
        frame[ymin:ymax, xmax, 1] = 0 
        frame[ymin:ymax, xmax, 2] = 1
        frame[ymin, xmin:xmax, 0] = 0 
        frame[ymin, xmin:xmax, 1] = 0 
        frame[ymin, xmin:xmax, 2] = 1
        frame[ymax, xmin:xmax, 0] = 0 
        frame[ymax, xmin:xmax, 1] = 0 
        frame[ymax, xmin:xmax, 2] = 1 
        plt.imshow(frame[...,::-1])
        plt.pause(0.0000001)



def plotabb(vid, xmin_in, xmax_in, ymin_in, ymax_in):
    # Play a video and plot multiple bounding boxes on it
    import matplotlib.pyplot as plt
    for frame_ind in range(len(np.array(vid).astype(int))):
        frame = np.array(vid).astype(int)[frame_ind]
        for bb_ind in range(xmin_in.shape[1]):
        
            xmin = xmin_in[frame_ind, bb_ind]
            xmax = xmax_in[frame_ind, bb_ind]
            ymin = ymin_in[frame_ind, bb_ind]
            ymax = ymax_in[frame_ind, bb_ind]
            if xmin!=-1:
                frame[ymin:ymax, xmin, 0] = 0
                frame[ymin:ymax, xmin, 1] = 0
                frame[ymin:ymax, xmin, 2] = 1
                frame[ymin:ymax, xmax, 0] = 0 
                frame[ymin:ymax, xmax, 1] = 0 
                frame[ymin:ymax, xmax, 2] = 1
                frame[ymin, xmin:xmax, 0] = 0 
                frame[ymin, xmin:xmax, 1] = 0 
                frame[ymin, xmin:xmax, 2] = 1
                frame[ymax, xmin:xmax, 0] = 0 
                frame[ymax, xmin:xmax, 1] = 0 
                frame[ymax, xmin:xmax, 2] = 1
        plt.imshow(frame[...,::-1])
        plt.pause(0.0000001)


def gt_maps_square(xmin, xmax, ymin, ymax, img_shape, labels, dims):
    # Draw a white square for each bounding box and blur it
    # Returns classes+1 outputs where the +1 is all classes put together
    output = np.zeros([dims+1]+list(img_shape))

    if (xmax[0,0]-xmin[0,0])*(ymax[0,0]-ymin[0,0]) < img_shape[1]*img_shape[2]/50.:
        sigma = 2
    else:
        sigma = 3
    for f_ind in range(xmin.shape[0]):
        for c_ind in range(xmin.shape[1]):
            cxmin = xmin[f_ind, c_ind]
            cxmax = xmax[f_ind, c_ind]
            cymin = ymin[f_ind, c_ind]
            cymax = ymax[f_ind, c_ind]
            output[labels[f_ind, c_ind], f_ind, cymin:cymax, cxmin:cxmax, :] = 1 
            output[dims, f_ind, cymin:cymax, cxmin:cxmax, :] = 1
        for d_ind in range(dims+1):
            output[d_ind, f_ind] = ndimage.gaussian_filter(output[d_ind, f_ind], sigma=(sigma), order=0)
    return output



def gt_maps_ellipse(xmin, xmax, ymin, ymax, img_shape, labels, dims):
    # Draw a white ellipse inside of each bounding box and blur it
    # Returns classes+1 outputs where the +1 is all classes put together

    output = np.zeros([dims+1]+list(img_shape))

    if (xmax[0,0]-xmin[0,0])*(ymax[0,0]-ymin[0,0]) < img_shape[1]*img_shape[2]/50.:
        sigma = 2
    else:
        sigma = 3

    for f_ind in range(xmin.shape[0]):
        for c_ind in range(xmin.shape[1]):
            emap = np.zeros(list(img_shape)[1:])
            cxmin = xmin[f_ind, c_ind]
            cxmax = xmax[f_ind, c_ind]
            cymin = ymin[f_ind, c_ind]
            cymax = ymax[f_ind, c_ind]

            center_x = int(cxmin+((cxmax-cxmin)/2.))
            center_y = int(cymin+((cymax-cymin)/2.))
            x_width = int((cxmax-cxmin)/2.)
            y_width = int((cymax-cymin)/2.)

            cv2.ellipse(emap, (center_x, center_y), (x_width, y_width), 0,0,360,[1,1,1],-1)

            output[labels[f_ind, c_ind], f_ind, :, :, :] = output[labels[f_ind, c_ind], f_ind, :, :, :]*(1-emap) + emap
            output[dims, f_ind, :, :, :] = output[dims, f_ind, :, :, :]*(1-emap) + emap

        for d_ind in range(dims+1):
            output[d_ind, f_ind] = ndimage.gaussian_filter(output[d_ind, f_ind], sigma=(sigma), order=0)
    return output








class Losses():
    def __init__(self, *args, **kwargs): #loss_type, size_average=None, reduce=None, reduction='mean', *args, **kwargs):
        """
        Args: 
            loss_type: String indicating which custom loss function is to be loaded.
        """

        self.loss_type = args[0]['loss_type']
        #self.loss_type = loss_type
        #self.size_average=size_average 
        #self.reduce=reduce
        #self.reduction=reduction

        self.loss_object = None

        if self.loss_type == 'HGC_MSE':
            self.loss_object = HGC_MSE(*args, **kwargs)

        elif self.loss_type == 'M_XENTROPY':
            self.loss_object = M_XENTROPY(*args, **kwargs)

        else:
            print('Invalid loss type selected. Quitting!')
            exit(1)

    def loss(self, predictions, data, **kwargs):
        """
        Args:
            predictions: Tensor output by the network
            target: Target tensor used with predictions to compute the loss
        """ 
        self.loss_object.loss(predictions, data, **kwargs)


class HGC_MSE():
    def __init__(self, *args, **kwargs):
        self.hgc_mse_loss = torch.nn.MSELoss() 

    def loss(self, predictions, data):
        bbox = data['bbox_data']
        targets = data['labels']
        play(np.array(data['data'][0].permute(1,2,3,0)))
        import pdb; pdb.set_trace()

        return self.hgc_mse_loss(predictions, targets)

    def gt_maps_square(xmin, xmax, ymin, ymax, img_shape, labels, dims):
        # Draw a white square for each bounding box and blur it
        # Returns classes+1 outputs where the +1 is all classes put together
        output = np.zeros([dims+1]+list(img_shape))
    
        if (xmax[0,0]-xmin[0,0])*(ymax[0,0]-ymin[0,0]) < img_shape[1]*img_shape[2]/50.:
            sigma = 2
        else:
            sigma = 3
        for f_ind in range(xmin.shape[0]):
            for c_ind in range(xmin.shape[1]):
                cxmin = xmin[f_ind, c_ind]
                cxmax = xmax[f_ind, c_ind]
                cymin = ymin[f_ind, c_ind]
                cymax = ymax[f_ind, c_ind]
                output[labels[f_ind, c_ind], f_ind, cymin:cymax, cxmin:cxmax, :] = 1 
                output[dims, f_ind, cymin:cymax, cxmin:cxmax, :] = 1
            for d_ind in range(dims+1):
                output[d_ind, f_ind] = ndimage.gaussian_filter(output[d_ind, f_ind], sigma=(sigma), order=0)
        return output

class M_XENTROPY():
    def __init__(self, *args, **kwargs):
        self.logsoftmax = nn.LogSoftmax()

    def loss(self, predictions, targets):
        targets = data['labels']
        return torch.mean(torch.sum(-targets * self.logsoftmax(predictions), dim=1))
