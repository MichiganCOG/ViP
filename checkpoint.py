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
 
def load_checkpoint(name, key_name='state_dict', remove_names=['module'], ignore_keys=[]):
    """
    Load checkpoint pickle file and return selected element from pickle file
    Args:
        name      (String): Full path, including pickle file name, to load 
        key_name  (String): Key name to return from saved pickle file 
        remove_names (List): List of names to remove from weights dict (e.g. module) 

    Return:
        Selected element from loaded checkpoint pickle file
    """
    checkpoint = torch.load(name)

    if key_name in checkpoint:
        return filter_weightst(checkpoint[key_name], remove_names, ignore_keys)
    else:
        return filter_weights(checkpoint, remove_names, ignore_keys)

def filter_weights(weight_dict, remove_names=['module'], ignore_keys=[]):
    dict_names = list(weight_dict.keys())
    #Example: module.base.12 -> base.12
    for name in dict_names:
        if name in ignore_keys:
            del weight_dict[name]
            continue 

        for rm in remove_names:
            if rm in name:
                new_name = '.'.join(name.split('.')[1:])
                weight_dict[new_name] = weight_dict[name]
                del weight_dict[name]

                if new_name in ignore_keys:
                    del weight_dict[new_name]

    return weight_dict
