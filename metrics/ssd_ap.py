import torch
from metrics.average_precision import AveragePrecision

class SSD_AP(AveragePrecision):
    """
    Compute Average Precision from the output of the SSD model
    Accumulates all predictions before computing AP
    """

    def __init__(self, threshold=0.5, num_points=11, *args, **kwargs):
        """
        Compute Average Precision (AP)
        Args:
            threshold    (float): iou threshold 
            num_points   (int): number of points to average for the interpolated AP calculation
            final_shape  (list) : [height, width] of input given to CNN
            result_dir   (String): save detections to this location
            ndata        (int): total number of datapoints in dataset 

        Return:
            None 
        """
        super(SSD_AP, self).__init__(threshold=threshold, num_points=num_points, *args, **kwargs)

    def get_accuracy(self, detections, data):
        """
        Args:
            detections (Tensor, shape [N,C,D,5]): predicted detections, each item [confidence, x1, y1, x2, y2]
            data:      (dictionary)
                - labels      (Tensor, shape [N,T,D_,5]):, each item [x1, y1, x2, y3, class] 
                - diff_labels (Tensor, shape [N,T,D_]):, difficult labels, each item (True or False)

        Return:
           Average Precision for SSD model  
        """

        gt     = data['labels'].squeeze(1)
        diff   = data['diff_labels'].squeeze(1)

        detections = detections.data
        N,C,D,_    = detections.shape
        _,D_,_     = gt.shape

        if self.count == 0:
            self.predictions = -1*torch.ones(self.ndata,C,D,5)
            self._targets    = -1*torch.ones(self.ndata,D_,5)
            self._diff       = torch.zeros(self.ndata,D_, dtype=torch.long)

        self.predictions[self.count:self.count+N] = detections * self.scale
        self._targets[self.count:self.count+N]    = gt
        self._diff[self.count:self.count+N]       = diff

        self.count += N

        #Only compute Average Precision after accumulating all predictions
        if self.count < self.ndata:
            return -1

        self.targets = -1*torch.ones(self.ndata,C,D_,4)
        for n, trgt in enumerate(self._targets):
            for d_ in range(D_):
                c = trgt[d_,-1].long() + 1 #c=0 is now the background class
                c = c * (1-self._diff[n,d_]) #skip difficult labels during calculation

                if c != 0:
                    self.targets[n,c,d_] = trgt[d_,:4]

        return self.get_AP(self.predictions, self.targets)
