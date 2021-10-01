import sys
import glob
import importlib

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
        metric_type = kwargs['acc_metric'] 
        self.metric_object = None

        metric_files = glob.glob('metrics/*.py')
        ignore_files = ['metrics.py']

        for mf in metric_files:
            if mf in ignore_files:
                continue

            module_name = mf[:-3].replace('/','.')
            module = importlib.import_module(module_name)
            module_lower = list(map(lambda module_x: module_x.lower(), dir(module)))

            if metric_type.lower() in module_lower:
                metric_index = module_lower.index(metric_type.lower())
                metric_function = getattr(module, dir(module)[metric_index])(**kwargs)

                self.metric_object = metric_function 

    def get_accuracy(self, predictions, targets, **kwargs):
        """
        Return accuracy from selected metric type

        Args:
            predictions: model predictions 
            targets: ground truth or targets 
        """

        if self.metric_object == None:
            return -1

        else:
            return self.metric_object.get_accuracy(predictions, targets, **kwargs)
