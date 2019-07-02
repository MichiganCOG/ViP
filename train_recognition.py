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
import datetime
import io
import yaml 
import torch
import torchvision
import numpy             as np
import torch.nn          as nn
import torch.optim       as optim
import torch.utils.data  as Data

#import models.models_import as models_import
#model = models_import.create_model_object(model_name='resnet101', num_classes=21, sample_size=224, sample_duration=16)
#import pdb; pdb.set_trace()
from parse_args                import Parse
from metrics                   import Metrics
from checkpoint                import save_checkpoint, load_checkpoint
#from torchvision               import datasets, transforms
from datasets                  import data_loader
from tensorboardX              import SummaryWriter
from torch.autograd            import Variable
from torch.optim.lr_scheduler  import MultiStepLR

# Import models 
from models.models_import      import create_model_object

def train(**args):

    print("\n############################################################################\n")
    print("Experimental Setup: ", args)
    print("\n############################################################################\n")

    avg_acc = []
    acc_metric = Metrics(args['acc_metric'])

    for total_iteration in range(args['rerun']):

        d = datetime.datetime.today()
        date = d.strftime('%Y%m%d-%H%M%S')
        result_dir = os.path.join(args['save_dir'], args['model'], '_'.join((args['dataset'],'[exp]',date)))
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
            train_loader = loader['train']
            valid_loader = loader['train'] # Run accuracy on train data if only `train` selected

        elif args['load_type'] == 'train_val':
            train_loader = loader['train']
            valid_loader  = loader['valid'] 

        else:
            print('Invalid environment selection for training, exiting')

        # END IF
    
        # Check if GPU is available (CUDA)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
        # Load Network # EDIT
        #model = create_model_object(model_name=args['model'],num_classes=args['labels'], sample_size=args['sample_size'], sample_duration=args['sample_duration']).to(device)
        model = create_model_object(**args).to(device)

        # Training Setup
        params     = [p for p in model.parameters() if p.requires_grad]

        if args['opt'] == 'sgd':
            optimizer  = optim.SGD(params, lr=args['lr'], momentum=args['momentum'], weight_decay=args['weight_decay'])

        elif args['opt'] == 'adam':
            optimizer  = optim.Adam(params, lr=args['lr'], weight_decay=args['weight_decay'])
        
        else:
            print('Unsupported optimizer selected. Exiting')
            exit(1)

        # END IF
            
        scheduler  = MultiStepLR(optimizer, milestones=args['milestones'], gamma=args['gamma'])    

    ############################################################################################################################################################################

        # Start: Training Loop
        for epoch in range(args['epoch']):
            running_loss = 0.0
            print('Epoch: ', epoch)

            # Setup Model To Train 
            model.train()

            import pdb; pdb.set_trace() 
            # Start: Epoch
            for step, data in enumerate(train_loader):
                # (True Batch, Augmented Batch, Sequence Length)
                x_input       = data['data'].to(device) 
                y_label       = data['labels'].to(device) 

                optimizer.zero_grad()

                outputs = model(x_input)

                # EDIT
                loss    = torch.mean(torch.sum(-y_label * nn.functional.log_softmax(outputs,dim=1), dim=1))
    
                loss.backward()
                optimizer.step()
    
                running_loss += loss.item()

                # Add Learning Rate Element
                for param_group in optimizer.param_groups:
                    writer.add_scalar(args['dataset']+'/'+args['model']+'/learningrate', param_group['lr'], epoch*len(train_loader) + step)

                # Add Loss Element
                writer.add_scalar(args['dataset']+'/'+args['model']+'/minibatchloss', loss.item(), epoch*len(train_loader) + step)

                if np.isnan(running_loss):
                    import pdb; pdb.set_trace()
   
                if (epoch*len(train_loader) + step) % 100 == 0:
                    print('Epoch: {}/{}, step: {}/{} | train loss: {:.4f}'.format(epoch, args['epoch'], step, len(train_loader), running_loss/100.))
                    running_loss = 0.0

                # END IF

            # Save Current Model
            save_path = os.path.join(save_dir, args['dataset']+'_epoch'+str(epoch)+'.pkl')
            save_checkpoint(epoch, 0, model, optimizer, save_path)
   
            # END FOR: Epoch

            scheduler.step()

            # Add Validation Accuracy 
            acc = 100.*valid(model, valid_loader, acc_metric, device)
            writer.add_scalar(args['dataset']+'/'+args['model']+'/validation_accuracy', acc, epoch)
 
            print('Accuracy of the network on the validation set: %d %%\n' % (acc))
    
        # END FOR: Training Loop

    ############################################################################################################################################################################

        # Close Tensorboard Element
        writer.close()

        # Save Final Model
        save_checkpoint(epoch + 1, 0, model, optimizer, args['save_dir']+'/'+str(total_iteration)+'/final_model.pkl')
        avg_acc.append(100.*valid(model, valid_loader, acc_metric, device))
    
    print("Average training accuracy across %d runs is %f" %(args['rerun'], np.mean(avg_acc)))

#Compute accuracy on validation split
def valid(model, valid_loader, acc_metric, device):

    model.eval()
    running_acc = []

    for step, data in enumerate(valid_loader):
        
        x_input = data['data'].to(device)
        y_label = data['labels'].to(device)

        outputs = model(x_input)
        
        running_acc.append(acc_metric.get_accuracy(outputs, y_label))

    return torch.mean(torch.Tensor(running_acc))

if __name__ == "__main__":

    parse = Parse()
    args = parse.get_args()

    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])

    train(**args)
