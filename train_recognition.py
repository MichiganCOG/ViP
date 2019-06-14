"""
LEGACY:
    View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
    My Youtube Channel: https://www.youtube.com/user/MorvanZhou
    Dependencies:
    torch: 0.4
    matplotlib
    numpy
"""
import os
import io
import cv2
import yaml
import torch
import torchvision
import numpy             as np
import torch.nn          as nn
import torch.optim       as optim
import torch.utils.data  as Data

import models.models_import as models_import
model = models_import.create_model_object(model_name='resnet_3d')
import pdb; pdb.set_trace()
#from utils                     import save_checkpoint, load_checkpoint, accuracy, accuracy_action
#from torchvision               import datasets, transforms
from datasets                  import data_loader
from tensorboardX              import SummaryWriter
from torch.autograd            import Variable
from torch.optim.lr_scheduler  import MultiStepLR

# Import models 
#from models                    import resnet18 as res



def train(args):

    print("Experimental Setup: ", args)

    avg_acc = []

    for total_iteration in range(args['Rerun']):

        # Tensorboard Element
        writer = SummaryWriter()

        # Load Data
        loader = data_loader(args)

        if args['Load_type'] == 'train':
            trainloader = loader['train']

        elif args['Load_type'] == 'train_val':
            trainloader = loader['train']
            testloader  = loader['valid'] 

        else:
            print('Invalid environment selection for training, exiting')

        # END IF
    
        # Check if GPU is available (CUDA)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        # Load Network # EDIT
        model = res(num_classes=args['Labels'], sample_size=args['Sample_size'], sample_duration=args['Sample_duration']).to(device)

        # Training Setup
        params     = [p for p in model.parameters() if p.requires_grad]

        if args['Opt'] == 'sgd':
            optimizer  = optim.SGD(params, lr=args['Lr'], momentum=args['Momentum'], weight_decay=args['Weight_decay'])

        elif args['Opt'] == 'adam':
            optimizer  = optim.Adam(params, lr=args['Lr'], weight_decay=args['Weight_decay'])
        
        else:
            print('Unsupported optimizer selected. Exiting')
            exit(1)

        # END IF
            
        scheduler  = MultiStepLR(optimizer, milestones=args['Milestones'], gamma=args['Gamma'])    

    ############################################################################################################################################################################

        # Start: Training Loop
        for epoch in range(args['Epoch']):
            running_loss = 0.0
            print('Epoch: ', epoch)

            # Save Current Model
            save_checkpoint(epoch, 0, model, optimizer, args['Save_dir']+'/'+str(total_iteration)+'/model_'+str(epoch)+'.pkl')

            # Setup Model To Train 
            model.train()

            # Start: Epoch
            for step, data in enumerate(trainloader):
    
                # (True Batch, Augmented Batch, Sequence Length)
                x_input       = data['data'].to(device) 
                y_label       = data['labels'].to(device) 

                optimizer.zero_grad()

                outputs = model(x_input)

                # EDIT
                loss    = torch.mean(torch.sum(-y_label * logsoftmax(outputs), dim=1))
    
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

                # END IF
   
            # END FOR: Epoch

            scheduler.step()

            acc = 100*accuracy_action(model, testloader, device)
            writer.add_scalar(args['Dataset']+'/'+args['Model']+'/train_accuracy', acc, epoch)
 
            print('Accuracy of the network on the training set: %d %%\n' % (acc))
    
        # END FOR: Training Loop

    ############################################################################################################################################################################

        # Close Tensorboard Element
        writer.close()

        # Save Final Model
        save_checkpoint(epoch + 1, 0, model, optimizer, args['Save_dir']+'/'+str(total_iteration)+'/final_model.pkl')
        avg_acc.append(100.*accuracy(model, testloader, device))
    
    print("Average training accuracy across %d runs is %f" %(args['Rerun'], np.mean(avg_acc)))


if __name__ == "__main__":

    with open('config.yaml', 'r') as filestream:
        args =  yaml.load(filestream)
    
    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args['Seed'])
    np.random.seed(args['Seed'])

    train(args)
