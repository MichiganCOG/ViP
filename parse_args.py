import argparse
import yaml

class Parse():

    def __init__(self):
        """
        Override config args with command-line args
        """
        
        parser = argparse.ArgumentParser()

        parser.add_argument('--cfg_file', type=str, default='config.yaml', help='Configuration file with experiment parameters')

        #Command-line arguments will override any config file arguments
        parser.add_argument('--rerun',           type=int, help='Number of trials to repeat an experiment')
        parser.add_argument('--dataset',         type=str, help='Name of dataset')
        parser.add_argument('--batch_size',      type=int, help='Numbers of videos in a mini-batch')
        parser.add_argument('--num_workers',     type=int, help='Number of subprocesses for dataloading')
        parser.add_argument('--load_type',       type=str, help='Environment selection, to include only training/training and validation/testing dataset')
        parser.add_argument('--model',           type=str, help='Name of model to be loaded')
        parser.add_argument('--labels',          type=int, help='Number of total classes in the dataset')
        parser.add_argument('--sample_size',     type=int, help='Height of frame to be provided as input to the model')
        parser.add_argument('--sample_duration', type=int, help='Temporal size of video to be provided as input to the model')

        parser.add_argument('--loss_type',    type=str,   help='Loss function')
        parser.add_argument('--acc_metric',   type=str,   help='Accuracy metric')
        parser.add_argument('--opt',          type=str,   help='Name of optimizer')
        parser.add_argument('--lr',           type=float, help='Learning rate')
        parser.add_argument('--momentum',     type=float, help='Momentum value in optimizer')
        parser.add_argument('--weight_decay', type=float, help='Weight decay')
        parser.add_argument('--milestones',   nargs='+',  help='Epoch values to change learning rate')
        parser.add_argument('--gamma',        type=float, help='Multiplier with which to change learning rate')
        parser.add_argument('--epoch',        type=int,   help='Total number of epochs')

        parser.add_argument('--json_path',    type=str, help='Path to train and test json files')
        parser.add_argument('--save_dir',     type=str, help='Path to results directory')
        parser.add_argument('--pretrained',   type=str, help='Load pretrained network')
        parser.add_argument('--subtract_mean',     type=str, help='Subtract mean (R,G,B) from all frames during preprocessing')
        parser.add_argument('--resize_shape', nargs=2,  help='(Height, Width) to resize original data')
        parser.add_argument('--final_shape',  nargs=2,  help='(Height, Width) of input to be given to CNN')
        parser.add_argument('--clip_length',  type=int, help='Number of frames within a clip')
        parser.add_argument('--clip_offset',  type=int, help='Frame offset between beginning of video and clip (1st clip only)')
        parser.add_argument('--clip_stride',  type=int, help='Frame offset between successive frames')
        parser.add_argument('--crop_shape',   nargs=2,  help='(Height, Width) of frame') 
        parser.add_argument('--crop_type',    type=str, help='Type of cropping operation (Random, Center and None)')
        parser.add_argument('--num_clips',    type=int, help='Number clips to be generated from a video (<0: uniform sampling, 0: Divide entire video into clips, >0: Defines number of clips)')

        parser.add_argument('--seed', type=int, help='Seed for reproducibility')

        #Dictionary of the command-line arguments passed
        self.cmd_args = vars(parser.parse_args()) 

        config_file = self.cmd_args['cfg_file']
        with open(config_file, 'r') as f:
            self.cfg_args = yaml.safe_load(f) #config file arguments

    def get_args(self):
        
        for (k,v) in self.cmd_args.items():
            if v is not None:
                self.cfg_args[k] = v

        return self.cfg_args
