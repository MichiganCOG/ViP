import torch 
import torch.nn    as nn


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
