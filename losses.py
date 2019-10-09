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

        reduction = 'mean' if 'reduction' not in kwargs else kwargs['reduction']
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

        return torch.mean(torch.sum(-one_hot * self.logsoftmax(predictions), dim=1))

class YC2BB_Attention_Loss(object):
    def __init__(self, *args, **kwargs):
       """
       Frame-wise attention loss used in:
       
       Weakly-supervised, no groundtruth labels are used.
       """

       self.loss_weighting = kwargs['has_loss_weighting']
       self.obj_interact   = kwargs['obj_interact']
       self.ranking_margin = kwargs['ranking_margin']
       self.loss_factor    = kwargs['loss_factor']

    def loss(self, predictions, data):
        """
        Args:
            predictions (List): 
                - output (): 
                - loss weighting():
            data        (NoneType)

        Return:
            Frame-wise weighting loss 
        """
        output, loss_weigh = predictions

        if self.loss_weighting or self.obj_interact: 
            rank_batch = F.margin_ranking_loss(output[:,0:1], output[:,1:2], 
                torch.ones(output.size()).type(output.data.type()), margin=self.ranking_margin, reduction='none')
            if self.loss_weighting and self.obj_interact:
                loss_weigh = (output[:, 0:1]+loss_weigh)/2. # avg
            elif self.loss_weighting:
                loss_weigh = output[:,0:1]
            else:
                loss_weigh = loss_weigh.unsqueeze(1)
            # ranking loss
            cls_loss = self.loss_factor*(rank_batch*loss_weigh).mean()+ \
                        (1-self.loss_factor)*-torch.log(2*loss_weigh).mean()
        else:
            # ranking loss
            cls_loss = F.margin_ranking_loss(output[:,0:1], output[:,1:2],
                torch.Tensor([[1],[1]]).type(output.data.type()), margin=self.ranking_margin)


        return cls_loss

