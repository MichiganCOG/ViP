import torch
import torch.nn as nn
import torch.nn.functional as F

#Code source: https://github.com/MichiganCOG/Video-Grounding-from-Text/blob/master/train.py
class YC2BB_Attention_Loss(object):
    def __init__(self, *args, **kwargs):
       """
       Frame-wise attention loss used in Weakly-Supervised Object Video Grounding... 
       https://arxiv.org/pdf/1805.02834.pdf
       
       Weakly-supervised, no groundtruth labels are used.
       """

       self.loss_weighting = kwargs['has_loss_weighting']
       self.obj_interact   = kwargs['obj_interact']
       self.ranking_margin = kwargs['ranking_margin']
       self.loss_factor    = kwargs['loss_factor']

    def loss(self, predictions, data):
        """
        Args:
            predictions (List): 
                - output (Tensor, shape [2*T, 2]): Positive and negative attention weights for each sample
                - loss_weigh (Tensor, shape [2*T, 1]): Loss weighting applied to each sampled frame
            data        (None) 

            T: number of sampled frames from video (default: 5)
        Return:
            Frame-wise weighting loss 
        """
        output, loss_weigh = predictions

        if self.loss_weighting or self.obj_interact:
            rank_batch = F.margin_ranking_loss(output[:,0:1], output[:,1:2],
                torch.ones(output.size()).type(output.data.type()), margin=self.ranking_margin, reduction='none')
            if self.loss_weighting and self.obj_interact:
                loss_weigh = (output[:, 0:1]+loss_weigh)/2. # avg
            elif self.loss_weighting:
                loss_weigh = output[:,0:1]
            else:
                loss_weigh = loss_weigh.unsqueeze(1)
            # ranking loss
            cls_loss = self.loss_factor*(rank_batch*loss_weigh).mean()+ \
                        (1-self.loss_factor)*-torch.log(2*loss_weigh).mean()
        else:
            # ranking loss
            cls_loss = F.margin_ranking_loss(output[:,0:1], output[:,1:2],
                torch.Tensor([[1],[1]]).type(output.data.type()), margin=self.ranking_margin)


        return cls_loss
