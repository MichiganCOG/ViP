import os
import gc
import sys 
import datetime
import yaml 
import torch
import psutil

import numpy             as np
import torch.nn          as nn
import torch.optim       as optim

from torch.optim.lr_scheduler           import MultiStepLR
from tensorboardX                       import SummaryWriter

from parse_args                         import Parse
from models.models_import               import create_model_object
from datasets.loading_function          import data_loader 
from losses                             import Losses
from metrics                            import Metrics
from checkpoint                         import save_checkpoint, load_checkpoint

def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())
    
def cpuStats():
        print(sys.version)
        print(psutil.cpu_percent())
        print(psutil.virtual_memory())  # physical memory usage
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
        print('memory GB:', memoryUse)


def train(**args):
    """
    Evaluate selected model 
    Args:
        rerun        (Int):        Integer indicating number of repetitions for the select experiment 
        seed         (Int):        Integer indicating set seed for random state
        save_dir     (String):     Top level directory to generate results folder
        model        (String):     Name of selected model 
        dataset      (String):     Name of selected dataset  
        exp          (String):     Name of experiment 
        debug        (Int):        Debug state to avoid saving variables 
        load_type    (String):     Keyword indicator to evaluate the testing or validation set
        pretrained   (Int/String): Int/String indicating loading of random, pretrained or saved weights
        opt          (String):     Int/String indicating loading of random, pretrained or saved weights
        lr           (Float):      Learning rate 
        momentum     (Float):      Momentum in optimizer 
        weight_decay (Float):      Weight_decay value 
        final_shape  ([Int, Int]): Shape of data when passed into network
        
    Return:
        None
    """

    print("\n############################################################################\n")
    print("Experimental Setup: ", args)
    print("\n############################################################################\n")

    for total_iteration in range(args['rerun']):

        # Generate Results Directory
        d          = datetime.datetime.today()
        date       = d.strftime('%Y%m%d-%H%M%S')
        result_dir = os.path.join(args['save_dir'], args['model'], '_'.join((args['dataset'],args['exp'],date)))
        log_dir    = os.path.join(result_dir,       'logs')
        save_dir   = os.path.join(result_dir,       'checkpoints')

        if not args['debug']:
            os.makedirs(result_dir, exist_ok=True)
            os.makedirs(log_dir,    exist_ok=True) 
            os.makedirs(save_dir,   exist_ok=True) 

            # Save Copy of Config File
            with open(os.path.join(result_dir, 'config.yaml'),'w') as outfile:
                yaml.dump(args, outfile, default_flow_style=False)


            # Tensorboard Element
            writer = SummaryWriter(log_dir)

        # Check if GPU is available (CUDA)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        # Load Network
        model = create_model_object(**args).to(device)

        # Load Data
        loader = data_loader(model_obj=model, **args)

        if args['load_type'] == 'train':
            train_loader = loader['train']
            valid_loader = loader['train'] # Run accuracy on train data if only `train` selected

        elif args['load_type'] == 'train_val':
            train_loader = loader['train']
            valid_loader = loader['valid'] 

        else:
            sys.exit('Invalid environment selection for training, exiting')

        # END IF
    
        # Training Setup
        params     = [p for p in model.parameters() if p.requires_grad]

        if args['opt'] == 'sgd':
            optimizer  = optim.SGD(params, lr=args['lr'], momentum=args['momentum'], weight_decay=args['weight_decay'])

        elif args['opt'] == 'adam':
            optimizer  = optim.Adam(params, lr=args['lr'], weight_decay=args['weight_decay'])
        
        else:
            sys.exit('Unsupported optimizer selected. Exiting')

        # END IF

        scheduler  = MultiStepLR(optimizer, milestones=args['milestones'], gamma=args['gamma'])

        if isinstance(args['pretrained'], str):
            ckpt = load_checkpoint(args['pretrained'])
            model.load_state_dict(ckpt)
            start_epoch = load_checkpoint(args['pretrained'], key_name='epoch') + 1
            optimizer.load_state_dict(load_checkpoint(args['pretrained'], key_name='optimizer'))

            for quick_looper in range(start_epoch):
                scheduler.step()

            # END FOR

        else:
            start_epoch = 0

        # END IF
            
        model_loss = Losses(device=device, **args)
        acc_metric = Metrics(**args)
        best_val_acc = 0.0

    ############################################################################################################################################################################


        # Start: Training Loop
        for epoch in range(start_epoch, args['epoch']):
            running_loss = 0.0
            print('Epoch: ', epoch)

            # Setup Model To Train 
            model.train()

            # Start: Epoch
            for step, data in enumerate(train_loader):
                if step% args['pseudo_batch_loop'] == 0:
                    loss = 0.0

                # END IF

                optimizer.zero_grad()
                x_input       = data['data'].to(device) 
                annotations   = data['annots'] 

                assert args['final_shape']==list(x_input.size()[-2:]), "Input to model does not match final_shape argument"

                #import pdb; pdb.set_trace()
                outputs = model(x_input)
                loss   = (1/args['pseudo_batch_loop'])*model_loss.loss(outputs, annotations)
                loss.backward()

                if step % args['pseudo_batch_loop'] == 0 and step > 0:
                    optimizer.step()

                    if not args['debug']:
                        # Add Learning Rate Element
                        for param_group in optimizer.param_groups:
                            writer.add_scalar(args['dataset']+'/'+args['model']+'/learning_rate', param_group['lr'], epoch*len(train_loader) + step)
                    
                        # Add Loss Element
                        writer.add_scalar(args['dataset']+'/'+args['model']+'/minibatch_loss', loss.item(), epoch*len(train_loader) + step)
                        if ((epoch*len(train_loader) + step+1) % 100 == 0):
                            print('Epoch: {}/{}, step: {}/{} | train loss: {:.4f}'.format(epoch, args['epoch'], step+1, len(train_loader), running_loss/float(step+1)))

                # END IF
    
                running_loss += loss.item()


                if np.isnan(running_loss):
                    import pdb; pdb.set_trace()

                # END IF

            # END FOR: Epoch

            if not args['debug']:
                # Save Current Model
                save_path = os.path.join(save_dir, args['dataset']+'_epoch'+str(epoch)+'.pkl')
                save_checkpoint(epoch, step, model, optimizer, save_path)
   
            # END IF: Debug

            scheduler.step()

            ## START FOR: Validation Accuracy
            running_acc = []
            running_acc = valid(valid_loader, running_acc, model, device, acc_metric)
            if not args['debug']:
                writer.add_scalar(args['dataset']+'/'+args['model']+'/validation_accuracy', 100.*running_acc[-1], epoch*len(valid_loader) + step)
            print('Accuracy of the network on the validation set: %f %%\n' % (100.*running_acc[-1]))

            # Save Best Validation Accuracy Model Separately
            if best_val_acc < running_acc[-1]:
                best_val_acc = running_acc[-1]

                if not args['debug']:
                    # Save Current Model
                    save_path = os.path.join(save_dir, args['dataset']+'_best_model.pkl')
                    save_checkpoint(epoch, step, model, optimizer, save_path)

                # END IF

            # END IF

        # END FOR: Training Loop

    ############################################################################################################################################################################

        if not args['debug']:
            # Close Tensorboard Element
            writer.close()

def valid(valid_loader, running_acc, model, device, acc_metric):
    model.eval()
    
    with torch.no_grad():
        for step, data in enumerate(valid_loader):
            x_input     = data['data'].to(device)
            annotations = data['annots'] 
            outputs     = model(x_input)
        
            running_acc.append(acc_metric.get_accuracy(outputs, annotations))
    
        # END FOR: Validation Accuracy

    return running_acc


if __name__ == "__main__":

    parse = Parse()
    args = parse.get_args()

    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args['seed'])
    #np.random.seed(args['seed']+1)

    train(**args)
