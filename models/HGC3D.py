import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import numpy as np

__all__ = [
    'HGC3D'
]
    

class DilatedConv3d(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size, kernel_depth, max_dilation, min_dilation, stride=1, padding=0, bias=None, stride_depth=1, depth_padding=0):
        super(DilatedConv3d, self).__init__()
        # This 3d convolution kernel is comprised of multiple 2d convolution kernels back to back
        # TODO need to work on implementing depthwise padding
        # TODO could also add depthwise dilation, but that is non-essential
        self.has_bias      = bias
        self.stride_depth  = stride_depth
        self.kernel_depth  = kernel_depth
        self.depth_padding = depth_padding
        self.convs         = nn.ModuleList() 

        # Uniformly step down the dilations from max_dilation to min_dilation from the beginning to the center of the video, then back up to max dilation at the end
        dilations = np.linspace(max_dilation, min_dilation,np.ceil((kernel_depth-1)/2.)).astype(int)

        if len(dilations)*2==kernel_depth:
            dilations = np.concatenate([dilations, dilations[::-1]], axis=0)
        else:
            dilations = np.concatenate([dilations, [min_dilation], dilations[::-1]], axis=0)

        
        init_zero = False 

        for frame in range(kernel_depth):
            current_dilation = dilations[frame] 
            current_padding  = int(padding + (((kernel_size-1)/2) * (current_dilation-1)))
            curr_conv = nn.Conv2d(in_chan, out_chan, kernel_size, stride=stride, padding=current_padding, dilation=current_dilation, bias=None)

            if init_zero:
                curr_conv.weight.data = nn.Parameter(torch.ones(curr_conv.weight.shape))
            else:
                nn.init.kaiming_uniform_(curr_conv.weight)

            self.convs.append(curr_conv)

        if self.has_bias==True:
            self.bias = nn.Parameter(torch.Tensor(out_chan))
            self.bias.requires_grad = True
            self.bias.data.zero_()
        

    def forward(self, x):

        # TODO This can be updated to incorporate a depthwise padding 
        batch, chan, frame, height, width = x.shape
        x_pad = nn.Parameter(x.new(batch, chan, frame+2*self.depth_padding, height, width).zero_())
        #x_pad[:, :, self.depth_padding:self.depth_padding+frame, :, :] = x  #### Causes Error:  leaf variable has been moved into the graph interior
        x_pad = x_pad.index_copy(2, x.new(range(self.depth_padding, self.depth_padding+frame)).long(), x)          
        x = x_pad


        vid_depth = x.shape[2]
        # The number of times the convolutions need to be performed across the depth of the video (depth = temporal dimension)
        block_starts = np.arange(0, vid_depth-self.kernel_depth+1, self.stride_depth)
        num_blocks = block_starts.shape[0]

        out = [] 
        
        for block_start in block_starts:
            out.append(self._single_block_convolution(x[:,:,block_start:block_start+self.kernel_depth,:,:]))

        out = torch.stack(out)
        # Remove empty dimension, replace with batch size
        out = out.permute(3,1,2,0,4,5)[0]
        
        if self.has_bias:
            out = out + self.bias.view(1,-1,1,1,1)
        
        return out


    def _single_block_convolution(self,x):
        assert (x.shape[2]==len(self.convs)), "Input must have same number of frames as depth of dilated conv3d"

        out = [] 
        for frame_id in range(x.shape[2]):
            frame = x[:,:,frame_id,:,:] # batch, in channel, depth, height, width
            #if self.convs[frame_id].weight.type() != x.type():
            #    self.convs[frame_id] = self.convs[frame_id].type(x.type())

            out.append(self.convs[frame_id](frame))

        out = torch.sum(torch.stack(out), dim=0)
        out = out.unsqueeze(2)

        return out

    def shape(self):
        inchan = self.convs[0].in_channels
        outchan = self.convs[0].out_channels
        kern = self.convs[0].kernel_size
        dil = self.convs[0].dilation
        depth = len(self.convs)

        # output channels, input channels, depth, kernel height, kernel width
        return (outchan, inchan, depth, (kern[0]-1)*dil[0]+1, (kern[1]-1)*dil[1]+1)



class HGC3D(nn.Module):

    def __init__(self,
                 sample_size,
                 sample_duration,
                 num_classes=400):
        self.inplanes = 64
        super(HGC3D, self).__init__()

        self.conv1 = DilatedConv3d(  3,  16, kernel_size=3, kernel_depth=3,  max_dilation=1, min_dilation=1, stride=1, padding=1, bias=True, stride_depth=1, depth_padding=1)
        self.conv1_bn = nn.BatchNorm3d(16)
        self.conv2 = DilatedConv3d( 16,  32, kernel_size=3, kernel_depth=5,  max_dilation=2, min_dilation=1, stride=1, padding=1, bias=True, stride_depth=1, depth_padding=2)
        self.conv2_bn = nn.BatchNorm3d(32)
        self.conv3 = DilatedConv3d( 32, 64, kernel_size=3, kernel_depth=5,  max_dilation=4, min_dilation=1, stride=1, padding=1, bias=True, stride_depth=1, depth_padding=0)
        self.conv3_bn = nn.BatchNorm3d(64)
        self.conv4 = DilatedConv3d(64, 128, kernel_size=5, kernel_depth=12, max_dilation=4, min_dilation=1, stride=1, padding=2, bias=True, stride_depth=1, depth_padding=0)
        self.conv4_bn = nn.BatchNorm3d(128)
        self.final = nn.Conv2d(128, num_classes*1+1, kernel_size=1, stride=1, padding=0, bias=None)
        #self.final = nn.Conv2d(256, num_classes*3+1, kernel_size=1, stride=1, padding=0, bias=None)

        self.idxtensor = torch.tensor([0])

        self.final_act = nn.Sigmoid()

    def forward(self, x):
        if x.is_cuda:
            self.conv1     = self.conv1.cuda()
            self.conv1_bn  = self.conv1_bn.cuda()
            self.conv2     = self.conv2.cuda()
            self.conv2_bn  = self.conv2_bn.cuda()
            self.conv3     = self.conv3.cuda()
            self.conv3_bn  = self.conv3_bn.cuda()
            self.conv4     = self.conv4.cuda()
            self.conv4_bn  = self.conv4_bn.cuda()
            self.final     = self.final.cuda()
            self.idxtensor = self.idxtensor.cuda()

        x = F.relu(self.conv1_bn(self.conv1(x)))
    
        x = F.relu(self.conv2_bn(self.conv2(x)))
     
        x = F.relu(self.conv3_bn(self.conv3(x)))
      
        x = F.relu(self.conv4_bn(self.conv4(x)))
       
        x = torch.index_select(x, 2, self.idxtensor).squeeze(2)
        x = self.final(x)
        x = self.final_act(x)

        return x





