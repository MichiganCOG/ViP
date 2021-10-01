import sys
import glob
import importlib

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

        loss_type   = kwargs['loss_type']
        self.loss_object = None
        
        loss_files = glob.glob('losses/*.py')
        ignore_files = ['losses.py']

        for lf in loss_files:
            if lf in ignore_files:
                continue

            module_name = lf[:-3].replace('/','.')
            module = importlib.import_module(module_name)
            module_lower = list(map(lambda module_x: module_x.lower(), dir(module)))

            if loss_type.lower() in module_lower:
                loss_index = module_lower.index(loss_type.lower())
                loss_function = getattr(module, dir(module)[loss_index])(**kwargs)

                self.loss_object = loss_function

        if self.loss_object is None:
            sys.exit('Loss function not found. Ensure .py file containing loss function is in losses/, with a matching class name')

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
