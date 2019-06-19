import os
import io #needed?
#import cv2 #needed?
import yaml
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import models.models_import as models_import
model = models_import.create_model_object(model_name='HGC3D', num_classes=21, sample_size=224, sample_duration=16)
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from datasets import data_loader 

def train(args):

    print('Experimental Setup: ',args)

    avg_acc = []

    for total_iteration in range(args['Rerun']):
        # Tensorboard Element
        writer = SummaryWriter()

        # Load Data
        loader = data_loader(args)#['Dataset'], args['Batch_size'], args['Load_type'])

        if args['Load_type'] == 'train':
            trainloader = loader['train']
        elif args['Load_type'] == 'train_val':
            trainloader = loader['train']
            testloader  = loader['valid'] 

        else:
            print('Invalid environment selection for training, exiting')

        # Check if GPU is available (CUDA)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        # Load Network # EDIT
        #model = res(num_classes=args['Labels'], sample_size=args['Sample_size'], sample_duration=args['Sample_duration']).to(device)

        # Training Setup
        params     = [p for p in model.parameters() if p.requires_grad]

        if args['Opt'] == 'sgd':
            optimizer  = optim.SGD(params, lr=args['Lr'], momentum=args['Momentum'], weight_decay=args['Weight_decay'])
        elif args['Opt'] == 'adam':
            optimizer  = optim.Adam(params, lr=args['Lr'], weight_decay=args['Weight_decay'])
        else:
            print('Unsupported optimizer selected. Exiting')
            exit(1)
            
        scheduler  = MultiStepLR(optimizer, milestones=args['Milestones'], gamma=args['Gamma'])    

        for epoch in range(args['Epoch']):
            running_loss = 0.0
            print('Epoch: ', epoch)

            # Save Current Model
            save_checkpoint(epoch, 0, model, optimizer, args['Save_dir']+'/'+str(total_iteration)+'/model_'+str(epoch)+'.pkl')

            # Setup Model To Train 
            model.train()

            for step, data in enumerate(trainloader):
                # (True Batch, Augmented Batch, Sequence Length)
                x_input       = data['data'].to(device) 
                y_label       = data['labels'].to(device) 

                optimizer.zero_grad()

                outputs = model(x_input)
                loss    = nn.functional.mse_loss(outputs, y_label) #TODO: Replace with Losses class
    
                loss.backward()
                optimizer.step()
    
                running_loss += loss.item()

                # Add Loss Element
                writer.add_scalar(args['Dataset']+'/'+args['Model']+'/loss', loss.item(), epoch*len(trainloader) + step)

                if np.isnan(running_loss):
                    import pdb; pdb.set_trace()
   
                if step % 100 == 0:
                    print('Epoch: ', epoch, '| train loss: %.4f' % (running_loss/100.))
                    running_loss = 0.0

            scheduler.step()

            acc = 100*accuracy_action(model, testloader, device)
            writer.add_scalar(args['Dataset']+'/'+args['Model']+'/train_accuracy', acc, epoch)
 
            print('Accuracy of the network on the training set: %d %%\n' % (acc))
    
        # Close Tensorboard Element
        writer.close()

        # Save Final Model
        save_checkpoint(epoch + 1, 0, model, optimizer, args['Save_dir']+'/'+str(total_iteration)+'/final_model.pkl')
        avg_acc.append(100.*accuracy(model, testloader, device))
    
    print("Average training accuracy across %d runs is %f" %(args['Rerun'], np.mean(avg_acc)))

if __name__ == '__main__':

    with open('config.yaml', 'r') as filestream:
        args = yaml.load(filestream)

    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args['Seed'])
    np.random.seed(args['Seed'])

    train(args)
