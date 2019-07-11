import os
import sys
import datetime
import yaml
import torch
import torchvision

import numpy                    as np
import torch.nn                 as nn
import torch.optim              as optim
import torch.utils.data         as Data

from torch.optim.lr_scheduler           import MultiStepLR
from tensorboardX                       import SummaryWriter

from parse_args                         import Parse
from models.models_import               import create_model_object
from datasets                           import data_loader 
from losses                             import Losses
from metrics                            import Metrics
from checkpoint                         import save_checkpoint, load_checkpoint

def train(**args):

    print('Experimental Setup: ',args)

    avg_acc = []

    for total_iteration in range(args['rerun']):
        d = datetime.datetime.today()
        date = d.strftime('%Y%m%d-%H%M%S')
        result_dir = os.path.join(args['save_dir'], args['model'], '_'.join((args['dataset'],args['exp'],date)))
        log_dir    = os.path.join(result_dir, 'logs')
        save_dir   = os.path.join(result_dir, 'checkpoints')

        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True) 
        os.makedirs(save_dir, exist_ok=True) 

        with open(os.path.join(result_dir, 'config.yaml'),'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)

        # Tensorboard Element
        writer = SummaryWriter(log_dir)

        # Load Data
        loader = data_loader(**args)

        if args['load_type'] == 'train':
            trainloader = loader['train']
        elif args['load_type'] == 'train_val':
            trainloader = loader['train']
            testloader  = loader['valid'] 

        else:
            sys.exit('Invalid environment selection for training, exiting')

        # Check if GPU is available (CUDA)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        # Load Network
        model = create_model_object(**args).to(device)
        if args['pretrained']:
                        model.load_state_dict(torch.load(args['pretrained']))

        # Training Setup
        params     = [p for p in model.parameters() if p.requires_grad]

        if args['opt'] == 'sgd':
            optimizer  = optim.SGD(params, lr=args['lr'], momentum=args['momentum'], weight_decay=args['weight_decay'])
        elif args['opt'] == 'adam':
            optimizer  = optim.Adam(params, lr=args['lr'], weight_decay=args['weight_decay'])
        else:
            sys.exit('Unsupported optimizer selected. Exiting')
            
        scheduler  = MultiStepLR(optimizer, milestones=args['milestones'], gamma=args['gamma'])    

        model_loss = Losses(device = device, **args)

        for epoch in range(args['epoch']):
            running_loss = 0.0
            print('Epoch: ', epoch)

            # Setup Model To Train 
            model.train()
            for step, data in enumerate(trainloader):
                # (True Batch, Augmented Batch, Sequence Length)
                data = dict((k, v.to(device)) for k,v in data.items())
                x_input       = data['data']
                y_label       = data['labels'] 

                optimizer.zero_grad()

                outputs = model(x_input)
                loss = model_loss.loss(outputs, data)

                loss.backward()
                optimizer.step()
    
                running_loss += loss.item()

                # Add Loss Element
                writer.add_scalar(args['dataset']+'/'+args['model']+'/loss', loss.item(), epoch*len(trainloader) + step)

                if np.isnan(running_loss):
                    import pdb; pdb.set_trace()
   
                if step % 100 == 0:
                    print('Epoch: ', epoch, '| train loss: %.4f' % (running_loss/100.))
                    running_loss = 0.0

            # Save Current Model
            save_path = os.path.join(save_dir, args['dataset']+'_epoch'+str(epoch)+'.pkl')
            save_checkpoint(epoch, 0, model, optimizer, save_path)

            scheduler.step()

            #acc = 100*accuracy_action(model, testloader, device)
            #writer.add_scalar(args['dataset']+'/'+args['model']+'/train_accuracy', acc, epoch)
 
            #print('Accuracy of the network on the training set: %d %%\n' % (acc))
    
        # Close Tensorboard Element
        writer.close()

        # Save Final Model
        save_checkpoint(epoch + 1, 0, model, optimizer, args['save_dir']+'/'+str(total_iteration)+'/final_model.pkl')
        #avg_acc.append(100.*accuracy(model, testloader, device))
    
    #print("Average training accuracy across %d runs is %f" %(args['rerun'], np.mean(avg_acc)))

if __name__ == '__main__':

    parse = Parse()
    args = parse.get_args()

    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])

    train(**args)
