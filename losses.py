import torch 
import torch.nn    as nn


class Losses():
    def __init__(self, loss_type, size_average=None, reduce=None, reduction='mean', *args, **kwargs):
        """
        Args: 
            loss_type: String indicating which custom loss function is to be loaded.
        """
        self.loss_type = loss_type
        self.size_average=size_average 
        self.reduce=reduce
        self.reduction=reduction

        self.loss_object = None

        if self.loss_type == 'HGC_MSE':
            self.loss_object = HGC_MSE(*args, **kwargs)

        elif self.loss_type == 'M_XENTROPY':
            self.loss_object = M_XENTROPY(*args, **kwargs)

        else:
            print('Invalid loss type selected. Quitting!')
            exit(1)

    def loss(self, predictions, targets, **kwargs):
        """
        Args:
            predictions: Tensor output by the network
            target: Target tensor used with predictions to compute the loss
        """ 
        self.loss_object.loss(predictions, targets, **kwargs)


class HGC_MSE():
    def __init__(self, *args, **kwargs):
        self.hgc_mse_loss = torch.nn.MSELoss() 

    def loss(self, predictions, targets):
        return self.hgc_mse_loss(predictions, targets)

class M_XENTROPY():
    def __init__(self, *args, **kwargs):
        self.logsoftmax = nn.LogSoftmax()

    def loss(self, predictions, targets):
        return torch.mean(torch.sum(-targets * self.logsoftmax(predictions), dim=1))
