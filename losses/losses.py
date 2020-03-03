import numpy as np
from scipy import ndimage
import os
import cv2

import torch 
import torch.nn as nn
import torch.nn.functional as F


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

        elif self.loss_type == 'YC2BB_Attention_Loss':
            self.loss_object = YC2BB_Attention_Loss(*args, **kwargs)

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
