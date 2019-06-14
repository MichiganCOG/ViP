'''
GOTURN (Generic Object Tracking Using Regression Networks) is siamese network that uses the first five convolutional layers of CaffeNet (AlexNet in this case). 
The two inputs of the network are the previous frame and the current frame, they both share convolutional layers. The features are then concatenated and passed through three fully connected layers
The output from the last fc layer are four numbers describing the bounding box position (x1, y1, x2, y2). Top left pixel coordinates and bottom right pixel coordinates.

Pretrained weights are transfered from Caffe using https://github.com/fanq15/caffe_to_torch_to_pytorch
.prototxt file from https://github.com/davheld/GOTURN
'''

import torch
import torch.nn as nn 
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models 

#Initialize weights of fully connected layers
def init_weights(m):
    if type(m) == nn.Linear:
        m.bias.data.fill_(1.0) #Initialize biases with 1
        m.weight.data.normal_(0, 0.005) #Initialize weights with normal distribution, zero mean and standard deviation of 0.005

#Local Response Normalization : basically just a contrast enchancement performed on the feature maps. Typically not used
#in anymore in favor or dropout and batch normalization (other regularization techniques)
#LRN code borrowed from: https://github.com/jiecaoyu/pytorch_imagenet/blob/master/networks/model_list/alexnet.py
class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=0.0001, beta=0.75, k=True):
        super(LRN, self).__init__()
        self.k = k 
        if self.k:
            self.average = nn.AvgPool3d(kernel_size=(local_size,1,1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.k:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x

class GoTurn(nn.Module):

    def __init__(self, train=True, dropout=0.5):
        super().__init__()

        #Generate features for both images individually
        self.features = nn.Sequential(
                nn.Conv2d(3,96,kernel_size=11,stride=4),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3,stride=2,padding=0,ceil_mode=True),
                LRN(local_size=5, alpha=0.0001, beta=0.75, k=True),
                nn.Conv2d(96,256,kernel_size=5,stride=1,padding=2,dilation=1,groups=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3,stride=2,padding=0,ceil_mode=True),
                LRN(local_size=5, alpha=0.0001, beta=0.75, k=True),
                nn.Conv2d(256,384,kernel_size=3,stride=1,padding=1),
                nn.ReLU(),
                nn.Conv2d(384,384,kernel_size=3,stride=1,padding=1,dilation=1,groups=2),
                nn.ReLU(),
                nn.Conv2d(384,256,kernel_size=3,stride=1,padding=1,dilation=1,groups=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3,stride=2,padding=0,ceil_mode=True))

        self.classifier = nn.Sequential(
                nn.Linear(256*6*6*2, 4096), #Two image inputs
                nn.ReLU(inplace = True),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace = True),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace = True),
                nn.Dropout(dropout),
                nn.Linear(4096, 4))

        if train:
            #Freeze conv layers
            for param in self.features.parameters():
                param.requires_grad = False

            #Re-initialize classification layers
            self.classifier.apply(init_weights)

    def forward(self,x0,x1):
        x0 = self.features(x0)
        x0 = x0.view(-1,256*6*6)
        x1 = self.features(x1)
        x1 = x1.view(-1,256*6*6)

        x = torch.cat((x0,x1),1)
        out = self.classifier(x) #output is in the format [x1,y1,x2,y2]

        #return out*10 #scale output by 10 per paper
        return out

