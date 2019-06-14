import torch 


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
