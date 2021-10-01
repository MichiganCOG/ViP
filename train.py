import os
import sys 
import datetime
import yaml 
import torch

import numpy             as np
import torch.nn          as nn
import torch.optim       as optim

from torch.optim.lr_scheduler           import MultiStepLR
from tensorboardX                       import SummaryWriter

from tools.parse_args                         import Parse
from models.models_import               import create_model_object
from datasets.loading_function          import data_loader 
from losses.losses                      import Losses
from metrics.metrics                    import Metrics
from tools.checkpoint                         import save_checkpoint, load_checkpoint

import pprint

import wandb

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

    print("Experimental Setup: ")
    pprint.PrettyPrinter(indent=4).pprint(args)

    for total_iteration in range(args['rerun']):

        # Generate Results Directory
        d          = datetime.datetime.today()
        date       = d.strftime('%Y%m%d-%H%M%S')
        result_dir = os.path.join(args['save_dir'], args['model'], '_'.join((args['dataset'],args['exp'],date)))
        log_dir    = os.path.join(result_dir,       'logs')
        save_dir   = os.path.join(result_dir,       'checkpoints')

        if not args['debug']:
            wandb.init(project=args['dataset'], name=args['exp'], config=args)

            #Replace result dir with wandb unique id, much easier to find checkpoints
            run_id = wandb.run.id 
            if run_id: 
                result_dir = os.path.join(args['save_dir'], args['model'], '_'.join((args['dataset'], run_id)))
                log_dir    = os.path.join(result_dir, 'logs')
                save_dir   = os.path.join(result_dir, 'checkpoints')

            os.makedirs(result_dir, exist_ok=True)
            os.makedirs(log_dir,    exist_ok=True) 
            os.makedirs(save_dir,   exist_ok=True) 

            # Save copy of config file
            with open(os.path.join(result_dir, 'config.yaml'),'w') as outfile:
                yaml.dump(args, outfile, default_flow_style=False)


            # Tensorboard Element
            writer = SummaryWriter(log_dir)

        # Check if GPU is available (CUDA)
        num_gpus = args['num_gpus']
        device = torch.device("cuda:0" if num_gpus > 0 and torch.cuda.is_available() else "cpu")
        print('Using {}'.format(device.type)) 

        # Load Network
        model = create_model_object(**args).to(device)
        model_obj = model 

        if device.type == 'cuda' and num_gpus > 1:
            device_ids = list(range(num_gpus)) #number of GPUs specified
            model = nn.DataParallel(model, device_ids=device_ids)
            model_obj = model.module #Model from DataParallel object has to be accessed through module

            print('GPUs Device IDs: {}'.format(device_ids))
    
        # Load Data
        loader = data_loader(model_obj=model_obj, **args)

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
            ckpt        = load_checkpoint(args['pretrained'], ignore_keys=args.get('ignore_ckpt_keys',[]))

            ckpt_keys = list(ckpt.keys())
            if ckpt_keys[0].startswith('module.'): #if checkpoint weights are from DataParallel object
                for key in ckpt_keys:
                    ckpt[key[7:]] = ckpt.pop(key)

            model_obj.load_state_dict(ckpt)

            if args['resume']:
                start_epoch = load_checkpoint(args['pretrained'], key_name='epoch') + 1

                optimizer.load_state_dict(load_checkpoint(args['pretrained'], key_name='optimizer'))
                #scheduler.step(epoch=start_epoch)

                for _ in range(start_epoch):
                    scheduler.step()

            else:
                start_epoch = 0

            # END IF 

        else:
            start_epoch = 0

        # END IF
            
        model_loss   = Losses(device=device, **args)
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
                    running_batch = 0
                    optimizer.zero_grad()

                # END IF

                x_input       = data['data'] 
                annotations   = data['annots']

                if isinstance(x_input, torch.Tensor):
                    mini_batch_size = x_input.shape[0]
                    outputs = model(x_input.to(device))

                    assert args['final_shape']==list(x_input.size()[-2:]), "Input to model does not match final_shape argument"
                else: #Model takes several inputs in forward function 
                    mini_batch_size = x_input[0].shape[0] #Assuming the first element contains the true data input 
                    for i, item in enumerate(x_input):
                        if isinstance(item, torch.Tensor):
                            x_input[i] = item.to(device)
                    outputs = model(*x_input)

                loss    = model_loss.loss(outputs, annotations)
                loss    = loss * mini_batch_size 
                loss.backward()

                running_loss  += loss.item()
                running_batch += mini_batch_size

                if np.isnan(running_loss):
                    import pdb; pdb.set_trace()

                # END IF

                if not args['debug']:
                    # Add Learning Rate Element
                    for param_group in optimizer.param_groups:
                        wandb.log({'lr':param_group['lr'],'train loss':loss.item()/mini_batch_size})
                        writer.add_scalar(args['dataset']+'/'+args['model']+'/learning_rate', param_group['lr'], epoch*len(train_loader) + step)

                    # END FOR
                
                    # Add Loss Element
                    writer.add_scalar(args['dataset']+'/'+args['model']+'/minibatch_loss', loss.item()/mini_batch_size, epoch*len(train_loader) + step)

                # END IF

                if ((epoch*len(train_loader) + step+1) % 100 == 0):
                    print('Epoch: {}/{}, step: {}/{} | train loss: {:.4f}'.format(epoch, args['epoch'], step+1, len(train_loader), running_loss/float(step+1)/mini_batch_size))

                # END IF

                if (epoch * len(train_loader) + (step+1)) % args['pseudo_batch_loop'] == 0 and step > 0:
                    # Apply large mini-batch normalization
                    for param in model.parameters():
                        if param.requires_grad:
                            param.grad *= 1./float(running_batch)

                    # END FOR
                    
                    # Apply gradient clipping
                    if ("grad_max_norm" in args) and float(args['grad_max_norm'] > 0):
                        nn.utils.clip_grad_norm_(model.parameters(),float(args['grad_max_norm']))

                    optimizer.step()
                    running_batch = 0

                # END IF
    

            # END FOR: Epoch
            
            scheduler.step()
            print('Schedulers lr: %f', scheduler.get_lr()[0])

            '''
            if not args['debug']:
                # Save Current Model
                save_path = os.path.join(save_dir, args['dataset']+'_epoch'+str(epoch)+'.pkl')
                save_checkpoint(epoch, step, model, optimizer, save_path)
            '''
   
            # END IF: Debug

            ## START FOR: Validation Accuracy
            running_acc = []
            running_acc = valid(valid_loader, running_acc, model, device)

            if not args['debug']:
                wandb.log({'epoch':epoch, 'val_accurracy':100*running_acc[-1]})
                writer.add_scalar(args['dataset']+'/'+args['model']+'/validation_accuracy', 100.*running_acc[-1], epoch*len(train_loader) + step)

            print('Accuracy of the network on the validation set: %f %%\n' % (100.*running_acc[-1]))

            # Save Best Validation Accuracy Model Separately
            if best_val_acc < running_acc[-1]:
                best_val_acc = running_acc[-1]

                if not args['debug']:
                    #Log best validation accuracy
                    wandb.run.summary['best_accuracy'] = best_val_acc

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

def valid(valid_loader, running_acc, model, device):
    acc_metric = Metrics(**args)
    model.eval()

    with torch.no_grad():
        for step, data in enumerate(valid_loader):
            x_input     = data['data']
            annotations = data['annots'] 

            if isinstance(x_input, torch.Tensor):
                outputs = model(x_input.to(device))
            else:
                for i, item in enumerate(x_input):
                    if isinstance(item, torch.Tensor):
                        x_input[i] = item.to(device)
                outputs = model(*x_input)
        
            running_acc.append(acc_metric.get_accuracy(outputs, annotations))

            if step % 100 == 0:
                print('Step: {}/{} | validation acc: {:.4f}'.format(step, len(valid_loader), running_acc[-1]))
    
        # END FOR: Validation Accuracy

    return running_acc


if __name__ == "__main__":

    parse = Parse()
    args = parse.get_args()

    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args['seed'])

    if not args['resume']:
        np.random.seed(args['seed'])

    train(**args)
