import torch

class MAP():

    def __init__(self, threshold=torch.linspace(0.5,0.95,10), num_points=101, *args, **kwargs):
        """
        (COCO) Mean average precision

        Args:
            threshold  (Tensor, shape[10]): Calculate AP at each of these threshold values
            num_points (float): number of points to average for the interpolated AP calculation
        """

        self.threshold = threshold
        self.IOU = IOU(average=False)
        self.AP = AveragePrecision(num_points=num_points, *args, **kwargs)

        self.result_dir = kwargs['result_dir']
        final_shape = kwargs['final_shape']
        #assuming model predictions are normalized between 0-1
        self.scale = torch.Tensor([1, final_shape[0], final_shape[1], final_shape[0], final_shape[1]]) #[1, height, width, height, width]

        self.ndata = kwargs['ndata']
        self.count = 0

    def get_mAP(self, predictions, targets):
        """
        Args:
            predictions (Tensor, shape [N,C,D,5]): prediction bounding boxes, coordinate format [confidence, x1, y1, x2, y2]
            targets (Tensor, shape [N,C,D_,4]): ground truth bounding boxes
            C:  num of classes + 1 (0th class is background class, not included in calculation)
            D:  predicted detections
            D_: ground truth detections 

        Return:
            Returns mAP score 
        """

        AP_scores = torch.zeros(self.threshold.shape)

        for n,t in enumerate(self.threshold):
            self.AP.update_threshold(t)
            AP_scores[n] = self.AP.get_AP(predictions, targets)

        return torch.mean(AP_scores)

    def get_accuracy(self, detections, data):
        """
        Args:
            detections (Tensor, shape [N,C,D,5]): predicted detections, each item [confidence, x1, y1, x2, y2]
            data:      (dictionary)
                - labels      (Tensor, shape [N,T,D_,5]):, each item [x1, y1, x2, y3, class] 

        Return:
            Returns mAP score 
        """
        gt     = data['labels'].squeeze(1)

        detections = detections.data
        N,C,D,_    = detections.shape
        _,D_,_     = gt.shape

        if self.count == 0:
            self.predictions = -1*torch.ones(self.ndata,C,D,5)
            self._targets    = -1*torch.ones(self.ndata,D_,5)

        self.predictions[self.count:self.count+N] = detections * self.scale
        self._targets[self.count:self.count+N]    = gt

        self.count += N

        #Only compute Mean Average Precision after accumulating all predictions
        if self.count < self.ndata:
            return -1

        self.targets = -1*torch.ones(self.ndata,C,D_,4)
        for n, trgt in enumerate(self._targets):
            for d_ in range(D_):
                c = trgt[d_,-1].long() + 1 #c=0 is now the background class

                if c != 0:
                    self.targets[n,c,d_] = trgt[d_,:4]

        return self.get_mAP(self.predictions, self.targets)
