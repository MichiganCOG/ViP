import os
import torch

def save_checkpoint(epoch, step, model, optimizer, save_path):
    """
    Save checkpoint pickle file with model weights and other experimental settings 
    Args:
        epoch     (Int):    Current epoch when model is being saved
        step      (Int):    Mini-batch iteration count when model is being saved
        model     (Object): Current copy of model
        optimizer (Object): Optimizer object
        save_path (String): Full directory path to results folder

    Return:
        None
    """

    state = {   'epoch':epoch,
                'step': step,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
             }

    torch.save(state, save_path)
 
def load_checkpoint(name, key_name='state_dict'):
    """
    Load checkpoint pickle file and return selected element from pickle file
    Args:
        name     (String): Full path, including pickle file name, to load 
        key_name (String): Key name to return from saved pickle file 

    Return:
        Selected element from loaded checkpoint pickle file
    """
    checkpoint = torch.load(name)

    if key_name not in checkpoint:
        return checkpoint

    return checkpoint[key_name]
