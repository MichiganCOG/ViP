import torch 


class Losses():
    def __init__(self, loss_type, size_average=None, reduce=None, reduction='mean'):
        self.loss_type = loss_type
        self.size_average=size_average 
        self.reduce=reduce
        self.reduction=reduction

        self.loss_object = None

        if self.loss_type == 'HGC_MSE':
            self.loss_object = HGC_MSE()


    def loss(self, predictions, targets):
        self.loss_object.loss(predictions, targets)


class HGC_MSE():
    def __init__(self):
        self.hgc_mse_loss = torch.nn.MSELoss() 

    def loss(self, predictions, targets):
        return self.hgc_mse_loss(predictions, targets)
