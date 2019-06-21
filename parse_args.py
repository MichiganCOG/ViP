import argparse
import yaml

class Parse():

    def __init__(self):
        """
        Override config args with command-line args
        """
        
        parser = argparse.ArgumentParser()

        parser.add_argument('--cfg_file', type=str, default='config.yaml', help='')

        parser.add_argument('--rerun', type=int, help='')
        parser.add_argument('--dataset', type=str, help='')
        parser.add_argument('--batch_size', type=int, help='')
        parser.add_argument('--load_type', type=str, help='')
        parser.add_argument('--model', type=str, help='')
        parser.add_argument('--labels', type=int, help='')
        parser.add_argument('--sample_size', type=int, help='')
        parser.add_argument('--sample_duration', type=int, help='')

        parser.add_argument('--opt', type=str, help='')
        parser.add_argument('--lr', type=float, help='')
        parser.add_argument('--momentum', type=float, help='')
        parser.add_argument('--weight_decay', type=float, help='')
        parser.add_argument('--milestones', type=int, help='') #TODO: verify data type, needs to be list of ints
        parser.add_argument('--gamma', type=float, help='')
        parser.add_argument('--epoch', type=int, help='')

        parser.add_argument('--save_dir', type=str, help='')
        parser.add_argument('--resize_shape', type=int, help='') #TODO: verify data type, needs to be list of ints
        parser.add_argument('--final_shape', type=int, help='') #TODO: verify data type, needs to be list of ints
        parser.add_argument('--clip_length', type=int, help='')
        parser.add_argument('--clip_offset', type=int, help='')
        parser.add_argument('--clip_stride', type=int, help='')
        parser.add_argument('--crop_shape', type=int, help='') #TODO: verify data type, needs to be list of ints
        parser.add_argument('--crop_type', type=str, help='')
        parser.add_argument('--num_clips', type=int, help='')

        parser.add_argument('--seed', type=int, help='')

        #Dictionary of the command-line arguments passed
        self.cmd_args = vars(parser.parse_args()) 

        config_file = self.cmd_args['cfg_file']
        with open(config_file, 'r') as f:
            self.cfg_args = yaml.load(f) #config file arguments

    def get_args(self):
        
        for (k,v) in self.cmd_args.items():
            if v is not None:
                self.cfg_args[k] = v

        return self.cfg_args
