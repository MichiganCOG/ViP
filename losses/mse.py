import torch

class MSE():
    def __init__(self, *args, **kwargs):
        """
        Mean squared error (squared L2 norm) between predictions and target

        Args:
            reduction (String): 'none', 'mean', 'sum' (see PyTorch Docs). Default: 'mean'
            device    (String): 'cpu' or 'cuda'

        Returns:
            None 
        """

        reduction = 'mean' if 'reduction' not in kwargs else kwargs['reduction']
        self.device = kwargs['device']

        self.mse_loss = torch.nn.MSELoss(reduction=reduction)

    def loss(self, predictions, data):
        """
        Args:
            predictions  (Tensor, shape [N,*]): Output by the network
            data         (dictionary)
                - labels (Tensor, shape [N,*]):  Targets from ground truth data

        Returns:
            Return mean squared error loss
        """

        targets = data['labels'].to(self.device)

        return self.mse_loss(predictions, targets)
