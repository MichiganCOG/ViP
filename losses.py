import torch 
import torch.nn    as nn
import numpy as np
from scipy import ndimage
import os
import cv2


class Losses(object):
    def __init__(self, *args, **kwargs): #loss_type, size_average=None, reduce=None, reduction='mean', *args, **kwargs):
        """
        Class used to initialize and handle all available loss types in ViP

        Args: 
            loss_type (String): String indicating which custom loss function is to be loaded.

        Return:
            Loss object 
        """

        self.loss_type   = kwargs['loss_type']
        self.loss_object = None
        
        if self.loss_type == 'MSE':
            self.loss_object = MSE(*args, **kwargs)

        elif self.loss_type == 'M_XENTROPY':
            self.loss_object = M_XENTROPY(*args, **kwargs)

        else:
            print('Invalid loss type selected. Quitting!')
            exit(1)

    def loss(self, predictions, data, **kwargs):
        """
        Function that calculates loss from selected loss type

        Args:
            predictions (Tensor, shape [N,*]): Tensor output by the network
            target      (Tensor, shape [N,*]): Target tensor used with predictions to compute the loss

        Returns:
            Calculated loss value
        """ 
        return self.loss_object.loss(predictions, data, **kwargs)

class MSE():
    def __init__(self, *args, **kwargs):
        """
        Mean squared error (squared L2 norm) between predictions and target

        Args:
            reduction (String): 'none', 'mean', 'sum' (see PyTorch Docs). Default: 'mean'
            device    (String): 'cpu' or 'cuda'

        Returns:
            None 
        """

        reduction = 'sum' if 'reduction' not in kwargs else kwargs['reduction']
        self.device = kwargs['device']

        self.mse_loss = torch.nn.MSELoss(reduction=reduction)

    def loss(self, predictions, data):
        """
        Args:
            predictions  (Tensor, shape [N,*]): Output by the network
            data         (dictionary)
                - labels (Tensor, shape [N,*]):  Targets from ground truth data

        Returns:
            Return mean squared error loss
        """

        targets = data['labels'].to(self.device)

        return self.mse_loss(predictions, targets)

class M_XENTROPY(object):
    def __init__(self, *args, **kwargs):
        """
        Cross-entropy Loss with a distribution of values, not just 1-hot vectors 

        Args:
            dim (integer): Dimension to reduce 

        Returns:
            None 
        """
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def loss(self, predictions, data):
        """
        Args:
            predictions  (Tensor, shape [N,*]): Output by the network
            data         (dictionary)
                - labels (Tensor, shape [N,*]):  Targets from ground truth data
                
        Return:
            Cross-entropy loss  
        """

        targets = data['labels']
        one_hot = np.zeros((targets.shape[0], predictions.shape[1]))
        one_hot[np.arange(targets.shape[0]), targets.cpu().numpy().astype('int32')[:, -1]] = 1
        one_hot = torch.Tensor(one_hot).cuda()

        return torch.sum(torch.sum(-one_hot * self.logsoftmax(predictions), dim=1))
