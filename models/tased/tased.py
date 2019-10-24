import torch
from torch import nn
import numpy                             as np
import torch.nn.functional               as F
import datasets.preprocessing_transforms as pt


"""
Code is from this Git Repository by  Kyle Min: https://github.com/MichiganCOG/TASED-Net 
Original TASED-Net paper: https://arxiv.org/abs/1908.05786

Testing:  This implementation provides accurate testing results given the pretrained model.
Training: This implementation does not alter the learning rate between the encoder and the decoder network which was done in the original paper.
"""


class TASED_v2(nn.Module):
    def __init__(self, **kwargs):
        super(TASED_v2, self).__init__()

        self.train_transforms = PreprocessTrain(**kwargs)
        self.test_transforms = PreprocessEval(**kwargs)

        self.base1 = nn.Sequential(
            SepConv3d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            BasicConv3d(64, 64, kernel_size=1, stride=1),
            SepConv3d(64, 192, kernel_size=3, stride=1, padding=1),
        )
        self.maxp2 = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.maxm2 = nn.MaxPool3d(kernel_size=(4,1,1), stride=(4,1,1), padding=(0,0,0))
        self.maxt2 = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), return_indices=True)
        self.base2 = nn.Sequential(
            Mixed_3b(),
            Mixed_3c(),
        )
        self.maxp3 = nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))
        self.maxm3 = nn.MaxPool3d(kernel_size=(4,1,1), stride=(4,1,1), padding=(0,0,0))
        self.maxt3 = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), return_indices=True)
        self.base3 = nn.Sequential(
            Mixed_4b(),
            Mixed_4c(),
            Mixed_4d(),
            Mixed_4e(),
            Mixed_4f(),
        )
        self.maxt4 = nn.MaxPool3d(kernel_size=(2,1,1), stride=(2,1,1), padding=(0,0,0))
        self.maxp4 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0), return_indices=True)
        self.base4 = nn.Sequential(
            Mixed_5b(),
            Mixed_5c(),
        )
        self.convtsp1 = nn.Sequential(
            nn.Conv3d(1024, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(1024, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),

            nn.ConvTranspose3d(1024, 832, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
            nn.BatchNorm3d(832, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),
        )
        self.unpool1 = nn.MaxUnpool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0))
        self.convtsp2 = nn.Sequential(
            nn.ConvTranspose3d(832, 480, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
            nn.BatchNorm3d(480, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),
        )
        self.unpool2 = nn.MaxUnpool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.convtsp3 = nn.Sequential(
            nn.ConvTranspose3d(480, 192, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
            nn.BatchNorm3d(192, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),
        )
        self.unpool3 = nn.MaxUnpool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.convtsp4 = nn.Sequential(
            nn.ConvTranspose3d(192, 64, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
            nn.BatchNorm3d(64, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),

            nn.Conv3d(64, 64, kernel_size=(2,1,1), stride=(2,1,1), bias=False),
            nn.BatchNorm3d(64, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),

            nn.ConvTranspose3d(64, 4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(4, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),

            nn.Conv3d(4, 4, kernel_size=(2,1,1), stride=(2,1,1), bias=False),
            nn.BatchNorm3d(4, eps=1e-3, momentum=0.001, affine=True),
            nn.ReLU(),

            nn.ConvTranspose3d(4, 4, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1), bias=False),
            nn.Conv3d(4, 1, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
        )

        if isinstance(kwargs['pretrained'], int) and kwargs['pretrained']:
            print('load pretrained')
            self._load_pretrained_weights()

    def _load_pretrained_weights(self):

        p_dict = torch.load('weights/TASED_v2.pt')
        s_dict = self.state_dict()
        for name in p_dict:
            val = p_dict[name]
            if 'module' in name:
                name = name[7:] 

            s_dict[name] = val 
        self.load_state_dict(s_dict)

    def forward(self, x):
        y3 = self.base1(x)
        y = self.maxp2(y3)
        y3 = self.maxm2(y3)
        _, i2 = self.maxt2(y3)
        y2 = self.base2(y)
        y = self.maxp3(y2)
        y2 = self.maxm3(y2)
        _, i1 = self.maxt3(y2)
        y1 = self.base3(y)
        y = self.maxt4(y1)
        y, i0 = self.maxp4(y)
        y0 = self.base4(y)

        z = self.convtsp1(y0)
        z = self.unpool1(z, i0)
        z = self.convtsp2(z)
        z = self.unpool2(z, i1, y2.size())
        z = self.convtsp3(z)
        z = self.unpool3(z, i2, y3.size())
        z = self.convtsp4(z)
        z = z.view(z.size(0), z.size(2), z.size(3), z.size(4))

        return z

class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SepConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(SepConv3d, self).__init__()
        self.conv_s = nn.Conv3d(in_planes, out_planes, kernel_size=(1,kernel_size,kernel_size), stride=(1,stride,stride), padding=(0,padding,padding), bias=False)
        self.bn_s = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu_s = nn.ReLU()

        self.conv_t = nn.Conv3d(out_planes, out_planes, kernel_size=(kernel_size,1,1), stride=(stride,1,1), padding=(padding,0,0), bias=False)
        self.bn_t = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu_t = nn.ReLU()

    def forward(self, x):
        x = self.conv_s(x)
        x = self.bn_s(x)
        x = self.relu_s(x)

        x = self.conv_t(x)
        x = self.bn_t(x)
        x = self.relu_t(x)
        return x

class Mixed_3b(nn.Module):
    def __init__(self):
        super(Mixed_3b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(192, 64, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(192, 96, kernel_size=1, stride=1),
            SepConv3d(96, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(192, 16, kernel_size=1, stride=1),
            SepConv3d(16, 32, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(192, 32, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)

        return out


class Mixed_3c(nn.Module):
    def __init__(self):
        super(Mixed_3c, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv3d(256, 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(256, 128, kernel_size=1, stride=1),
            SepConv3d(128, 192, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(256, 32, kernel_size=1, stride=1),
            SepConv3d(32, 96, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(256, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4b(nn.Module):
    def __init__(self):
        super(Mixed_4b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(480, 192, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(480, 96, kernel_size=1, stride=1),
            SepConv3d(96, 208, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(480, 16, kernel_size=1, stride=1),
            SepConv3d(16, 48, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(480, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4c(nn.Module):
    def __init__(self):
        super(Mixed_4c, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 160, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 112, kernel_size=1, stride=1),
            SepConv3d(112, 224, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 24, kernel_size=1, stride=1),
            SepConv3d(24, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4d(nn.Module):
    def __init__(self):
        super(Mixed_4d, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 128, kernel_size=1, stride=1),
            SepConv3d(128, 256, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 24, kernel_size=1, stride=1),
            SepConv3d(24, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4e(nn.Module):
    def __init__(self):
        super(Mixed_4e, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 112, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 144, kernel_size=1, stride=1),
            SepConv3d(144, 288, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 32, kernel_size=1, stride=1),
            SepConv3d(32, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4f(nn.Module):
    def __init__(self):
        super(Mixed_4f, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(528, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(528, 160, kernel_size=1, stride=1),
            SepConv3d(160, 320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(528, 32, kernel_size=1, stride=1),
            SepConv3d(32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(528, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_5b(nn.Module):
    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(832, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(832, 160, kernel_size=1, stride=1),
            SepConv3d(160, 320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(832, 32, kernel_size=1, stride=1),
            SepConv3d(32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(832, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_5c(nn.Module):
    def __init__(self):
        super(Mixed_5c, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(832, 384, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(832, 192, kernel_size=1, stride=1),
            SepConv3d(192, 384, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(832, 48, kernel_size=1, stride=1),
            SepConv3d(48, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(832, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out



class PreprocessTrain(object):
    """
    Container for all transforms used to preprocess clips for training in this dataset.
    """
    def __init__(self, **kwargs):
        """
        Initialize preprocessing class for training set
        Args:
            preprocess (String): Keyword to select different preprocessing types            
            crop_type  (String): Select random or central crop 

        Return:
            None
        """

        self.transforms  = []
        self.transforms1 = []
        self.preprocess  = kwargs['preprocess']
        crop_type        = kwargs['crop_type']


        self.transforms.append(pt.ResizeClip(**kwargs))
        

        self.transforms.append(pt.RandomFlipClip(direction='h', p=0.5, **kwargs))
        self.transforms.append(pt.ToTensorClip(**kwargs))

    def __call__(self, input_data):
        for transform in self.transforms:
            input_data = transform(input_data)

        # Force data between -1 and 1
        input_data = input_data*2-1

        return input_data


class PreprocessEval(object):
    """
    Container for all transforms used to preprocess clips for training in this dataset.
    """
    def __init__(self, **kwargs):
        """
        Initialize preprocessing class for training set
        Args:
            preprocess (String): Keyword to select different preprocessing types            
            crop_type  (String): Select random or central crop 

        Return:
            None
        """

        self.transforms = []

        self.transforms.append(pt.ResizeClip(**kwargs))
        self.transforms.append(pt.ToTensorClip(**kwargs))


    def __call__(self, input_data):
        for transform in self.transforms:
            input_data = transform(input_data)


        # Force data between -1 and 1
        input_data = input_data*2-1

        return input_data


