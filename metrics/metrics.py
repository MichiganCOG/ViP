import os
import json 
import numpy as np

import torch

class Metrics(object):
    def __init__(self, *args, **kwargs):
        """
        Compute accuracy metrics from this Metrics class
        Args:
            acc_metric (String): String used to indicate selected accuracy metric 
    
        Return:
            None
        """
        self.metric_type = kwargs['acc_metric'] 

        if self.metric_type == 'Accuracy':
            self.metric_object = Accuracy(*args, **kwargs) 
        elif self.metric_type == 'AveragePrecision':
            self.metric_object = AveragePrecision(*args, **kwargs)
        elif self.metric_type == 'mAP':
            self.metric_object = MAP(*args, **kwargs)
        elif self.metric_type == 'SSD_AP':
            self.metric_object = SSD_AP(*args, **kwargs)
        elif self.metric_type == 'Box_Accuracy':
            self.metric_object = Box_Accuracy(*args, **kwargs)
        else:
            self.metric_type = None

    def get_accuracy(self, predictions, targets, **kwargs):
        """
        Return accuracy from selected metric type

        Args:
            predictions: model predictions 
            targets: ground truth or targets 
        """

        if self.metric_type == None:
            return -1

        else:
            return self.metric_object.get_accuracy(predictions, targets, **kwargs)
