import os
import torch

def save_checkpoint(epoch, step, model, optimizer, filename):
    state = {   'epoch':epoch,
                'step': step,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
             }

    path = os.path.join('results/', filename)

    torch.save(state, path)
   

 
def load_checkpoint(name, key_name='state_dict'):
    checkpoint = torch.load(os.path.join('results',name))
    return checkpoint[key_name]
