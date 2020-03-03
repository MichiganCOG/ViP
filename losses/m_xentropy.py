import torch
import numpy as np

class M_XENTROPY(object):
    def __init__(self, *args, **kwargs):
        """
        Cross-entropy Loss with a distribution of values, not just 1-hot vectors 

        Args:
            dim (integer): Dimension to reduce 

        Returns:
            None 
        """
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def loss(self, predictions, data):
        """
        Args:
            predictions  (Tensor, shape [N,*]): Output by the network
            data         (dictionary)
                - labels (Tensor, shape [N,*]):  Targets from ground truth data
                
        Return:
            Cross-entropy loss  
        """

        targets = data['labels']
        one_hot = np.zeros((targets.shape[0], predictions.shape[1]))
        one_hot[np.arange(targets.shape[0]), targets.cpu().numpy().astype('int32')[:, -1]] = 1
        one_hot = torch.Tensor(one_hot).cuda()

        return torch.mean(torch.sum(-one_hot * self.logsoftmax(predictions), dim=1))

