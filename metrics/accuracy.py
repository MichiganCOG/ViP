import torch
import numpy as np

class Accuracy(object):
    """
    Standard accuracy computation. # of correct cases/# of total cases

    """
    def __init__(self, *args, **kwargs):
        self.correct = 0.
        self.total   = 0. 

    def get_accuracy(self, predictions, data):
        """
        Args:
            predictions (Tensor, shape [N,*])
            data        (dictionary):
                - labels (Tensor, shape [N,*]) 

        Return:
            Accuracy # of correct case/ # of total cases
        """
        targets = data['labels']
        assert (predictions.shape[0] == targets.shape[0])

        targets     = targets.detach().cpu().numpy()
        predictions = predictions.detach().cpu().numpy()

        if len(targets.shape) == 2 and len(predictions.shape) == 2:
            self.correct += np.sum(np.argmax(predictions,1) == targets[:, -1])
            self.total   += predictions.shape[0]

        else: 
            self.correct += np.sum(np.argmax(predictions,1) == targets[:, -1])
            self.total   += predictions.shape[0]

        # END IF

        return self.correct/self.total

