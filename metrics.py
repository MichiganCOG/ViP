import torch


class Metrics():
    def __init__(self, *args, **kwargs):
        """
        Compute accuracy metrics from this Metrics class

        """
        self.metric_type = kwargs['acc_metric'] 

        if self.metric_type == 'Accuracy':
            self.metric_object = Accuracy(*args, **kwargs) 
        elif self.metric_type == 'IOU':
            self.metric_object = IOU(*args, **kwargs)
        elif self.metric_type == 'Precision':
            self.metric_object = Precision(*args, **kwargs) 
        elif self.metric_type =='AP':
            self.metric_object = AveragePrecision(*args, **kwargs)
        elif self.metric_type == 'SSD_AP':
            self.metric_object = SSD_AP(*args, **kwargs)
        elif self.metric_type == 'mAP':
            self.metric_object = MAP(*args, **kwargs)
        elif self.metric_type == 'Recall':
            self.metric_object = Recall(*args, **kwargs) 
        elif self.metric_type == 'AR':
            self.metric_object = AverageRecall(*args, **kwargs)
        else:
            self.metric_type = None

    def get_accuracy(self, predictions, targets, **kwargs):
        """
        Return accuracy from selected metric type

        Args:
            predictions: model predictions 
            targets: ground truth or targets 
        """

        return self.metric_object.get_accuracy(predictions, targets, **kwargs)

class Accuracy():
    """
    Standard accuracy computation. # of correct cases/# of total cases

    """
    def __init__(self, *args, **kwargs):
        pass

    def get_accuracy(self, predictions, targets):
        
        assert (predictions.shape == targets.shape)

        correct = torch.sum(predictions == targets).float()
        total = predictions.nelement()

        return correct/total

class IOU():
    """
    Intersection-over-union between two bounding boxes

    Args:
        average: if False, return all iou values rather than the arthimetic mean 
    """
    def __init__(self, average=True, *args, **kwargs):
        
        self.average = average

    def intersect(self, box_p, box_t):
        """
        Intersection area between both bounding boxes

        Args:
            box_p: prediction bounding box, shape [N,4]
            box_t: target bounding box, shape [N,4]

        Return:
            intersect area, shape [N,1]
        """

        try :
            assert torch.all(box_p[:,0] < box_p[:,2])
            assert torch.all(box_p[:,1] < box_p[:,3])
            assert torch.all(box_t[:,0] < box_t[:,2])
            assert torch.all(box_t[:,1] < box_t[:,3])
        except AssertionError:
            return torch.Tensor([0])

        x_left = torch.max(box_p[:,0], box_t[:,0])
        y_top = torch.max(box_p[:,1], box_t[:,1])
        x_right = torch.min(box_p[:,2], box_t[:,2])
        y_bottom = torch.min(box_p[:,3], box_t[:,3])

        intersect_area = (x_right - x_left) * (y_bottom - y_top)

        return torch.clamp(intersect_area, min=0)

    def iou(self, box_p, box_t):
        """
        Args:
            box_p: prediction bounding box, shape [N,4], coordinate format [x1, y1, x2, y2]
            box_t: target bounding box, shape [N,4]

        Return:
            iou: shape [N,1]
        """
        
        intersect_area = self.intersect(box_p, box_t)

        if intersect_area == 0:
            return torch.Tensor([0])

        box_p_area = (box_p[:,2] - box_p[:,0]) * (box_p[:,3] - box_p[:,1])
        box_t_area = (box_t[:,2] - box_t[:,0]) * (box_t[:,3] - box_t[:,1])
        union = box_p_area + box_t_area - intersect_area 
        
        return intersect_area / union 

    def get_accuracy(self, predictions, targets):
        """
        Args:
            predictions: shape [N] or [N,C,4], coordinate format [x1, y1, x2, y2]
            targets: shape [N,4] or [N,C,4]

        Return:
            iou: scalar or shape [N,C]
        """

        if len(predictions.shape) > 2:
            n,c,_ = predictions.shape
            iou_scores = torch.zeros((n,c))

            assert c == targets.shape[1]
            
            for cls in range(c):
                iou_scores[:,cls] = self.iou(predictions[:,cls,:],targets[:,cls,:])
        else:
            iou_scores = self.iou(predictions, targets)

        if self.average:
            return torch.mean(iou_scores)
        else:
            return iou_scores

class Precision():

    def __init__(self, threshold=0.5, *args, **kwargs):

        self.threshold = threshold
        self.IOU = IOU(average=False)

    def get_precision(self, scores, targets_mask):
        """
        Args:
            scores: confidence score (or iou) per prediction, shape [N] 
            targets_mask: binary mask, shape [N]
        """

        TP = torch.sum((scores * targets_mask) >= self.threshold).float()
        FP = torch.sum((scores * (1-targets_mask) >= self.threshold)).float()

        return TP/(TP+FP)

    def get_accuracy(self, predictions, targets):
        """
        Args:
            predictions: shape [N,4], coordinate format [x1, y1, x2, y2]
            targets: shape [N,4]
        """

        n,_ = targets.shape 
        targets_mask = torch.ones(n)
        scores = self.IOU.get_accuracy(predictions, targets) #TODO: Find alternate way to compute scores 

        return self.get_precision(scores, targets_mask)

class AveragePrecision():

    def __init__(self, threshold=0.5, num_points=101, *args, **kwargs):
        """
        Compute Average Precision (AP)
        Args:
            threshold: (scalar) 
            num_points: number of points to average for the interpolated AP calculation
        """

        self.threshold = threshold 
        self.num_points = num_points
        self.IOU = IOU(average=False)

    def update_threshold(self, threshold):
        self.threshold = threshold

    def get_average_precision(self, scores, targets_mask):
        """
        Args:
            scores: confidence score (or iou) per prediction, shape [N] 
            targets_mask: binary mask, shape [N,C]
        """

        scores, idx = torch.sort(scores, descending=True)
        targets_mask = targets_mask[:,idx]

        pr = torch.zeros(len(scores))
        rc = torch.zeros(len(scores))

        running_tp = 0 #running total number of predicted true positives
        total_positives = torch.sum(targets_mask)

        #Get values for precision-recall curve
        for n,(s,t) in enumerate(zip(scores, targets_mask)):
            tp = torch.sum((s * t) >= self.threshold).float() #true positive

            running_tp += tp 
            pr[n] = running_tp/(n+1)
            rc[n] = running_tp/total_positives

        pr_inter = torch.zeros(self.num_points) #interpolated n-point precision curve
        rc_values = torch.linspace(0,1,self.num_points) #sampled recall points for precision-recall curve

        #The interpotaled P-R curve will take on the max precision value to the right at each recall
        for n in range(len(rc_values)):
            idx = rc >= rc_values[n]

            if len(pr[idx]) == 0:
                pr_inter[n] = 0
            else:
                pr_inter[n] = max(pr[idx])

        return torch.mean(pr_inter)
    
    def get_accuracy(self, predictions, targets):
        """
        Args:
            predictions: shape [N,C,4], coordinate format [x1, y1, x2, y2]
            targets: shape [N,C,4]
        """

        if len(targets.shape) > 2:
            n,c,_ = targets.shape
            targets_mask = torch.zeros(n,c)
            for cls in range(c):
                targets_mask[0,cls] = 1 - torch.equal(targets[0,cls], torch.Tensor([-1,-1,-1,-1]))

            #TODO: Find way to generate masks for multiple objects. None relevant entries should be coordinates of -1
        else: #single class case
            n,_ = targets.shape 
            targets_mask = 1 - (targets == torch.Tensor([-1,-1,-1,-1]))[:,0].float()

        scores = self.IOU.get_accuracy(predictions, targets) #[N,C] 

        return self.get_average_precision(scores, targets_mask)

class SSD_AP(AveragePrecision):
    """
    Compute Average Precision from the output of the SSD model
    Accumulates all predictions before computing AP
    """

    def __init__(self, threshold=0.5, num_points=11, *args, **kwargs):
        """
        Compute Average Precision (AP)
        Args:
            threshold: (scalar) 
            num_points: number of points to average for the interpolated AP calculation
        """
        super(SSD_AP, self).__init__(threshold=threshold, num_points=num_points)

        self.result_dir = kwargs['result_dir']
        resize_shape = kwargs['resize_shape']
        #self.scale = torch.Tensor([resize_shape[0], resize_shape[1], resize_shape[0], resize_shape[1]]) #to scale predictions to image size
        self.scale = torch.Tensor([353, 500, 353, 500])

        self.ndata = kwargs['ndata']
        self.count = 0

    def get_accuracy(self, detections, gt):
        """
        Args:
            detections: shape [N,C,D,5], each item [x1, y1, x2, y2, confidence]
            gt: shape [N,D_,5], each item [x1, y1, x2, y3, class] 
        """

        detections = detections.data
        N,C,D,_ = detections.shape
        _,D_,_ = gt.shape 

        if self.count == 0:
            self._predictions = -1*torch.ones(self.ndata,C,D,5).to(detections.device)
            self._targets = -1*torch.ones(self.ndata,D_,5).to(detections.device)

        self._predictions[self.count:self.count+N] = detections 
        self._targets[self.count:self.count+N] = gt

        self.count += N

        #Only compute Average Precision after accumulating all predictions
        if self.count < self.ndata:
            return -1

        self.predictions = -1*torch.ones(self.ndata,C,D,4).to(detections.data)
        self.targets = -1*torch.ones(self.ndata,C,D_,4).to(detections.data)
        
        #Retain only predictions with high confidence above threshold
        for n, pred in enumerate(self._predictions):
            for c in range(C):
                d = 0

                while pred[c, d, 0] >= 0.6:
                    pt = (pred[c, d, 1:]*self.scale).cpu().numpy()
                    self.predictions[n,c,d] = torch.Tensor(pt).to(detections.device)

                    d += 1

        for n, trgt in enumerate(self._targets):
            for d_ in range(gt.shape[1]):
                c = trgt[d_, -1].long() + 1 #c=0 is now the background class
                self.targets[n,c,0] = trgt[d_,:4]

        import pdb; pdb.set_trace()
        return super(SSD_AP,self).get_accuracy(self.predictions, self.targets)

class MAP():

    def __init__(self, threshold=torch.linspace(0.5,0.95,10), num_points=101, *args, **kwargs):
        """
        Mean average precision

        Args:
            threshold: Calculate AP at each of these threshold values
            num_points: number of points to average for the interpolated AP calculation
        """

        self.threshold = threshold
        self.IOU = IOU(average=False)
        self.AP = AveragePrecision(num_points=num_points)

    def get_mAP(self, predictions, targets):
        """
        Args:
            predictions: shape [N,C,4], coordinate format [x1, y1, x2, y2]
            targets: shape [N,C,4]
        """

        AP_scores = torch.zeros(self.threshold.shape)

        for n,t in enumerate(self.threshold):
            self.AP.update_threshold(t)
            AP_scores[n] = self.AP.get_accuracy(predictions, targets)

        return torch.mean(AP_scores)

    def get_accuracy(self, predictions, targets):
        """
        Args:
            predictions: shape [N,C,4], coordinate format [x1, y1, x2, y2]
            targets: shape [N,C,4]
        """

        return self.get_mAP(predictions, targets)

class Recall():
    def __init__(self, threshold=0.5, *args, **kwargs):

        self.threshold = threshold
        self.IOU = IOU(average=False)

    def get_recall(self, scores, targets_mask):
        """
        Args:
            scores: confidence scores per prediction, shape [N]
            targets_mask: binary mask, shape [N]
        """
        TP = torch.sum((scores * targets_mask) >= self.threshold).float()
        FN = torch.sum((scores * targets_mask) < self.threshold).float()

        return TP/(TP+FN)

    def get_accuracy(self, predictions, targets):
        """
        Args:
            predictions: shape [N,4], coordinate format [x1, y1, x2, y2]
            targets: shape [N,4]
        """
        n,c,_ = targets.shape 
        targets_mask = torch.ones((n,c))
        scores = self.IOU.get_accuracy(predictions, targets) #TODO: Add alternate way to compute scores

        return self.get_recall(scores, targets_mask)

class AverageRecall():
    #TODO: Incomplete
    def __init__(self, threshold=0.5, det=None, *args, **kwargs):
        """
        Compute Average Recall (AR)

        Args:
            threshold: (scalar)
            det: max number of detections per image (optional)
        """
        
        self.threshold = threshold
        self.det = det
        self.IOU = IOU()

    def get_recall(self, predictions, targets, targets_mask):
        """
        Args:
            predictions: shape [N,C,4], coordinate format [x1, y1, x2, y2]
            targets: shape [N,C,4]
            targets_mask: binary mask, shape [N,C]
        """
        iou_values = self.IOU.get_accuracy(predictions, targets) #[N,C] 

        TP = torch.sum((iou_values * targets_mask) >= self.threshold).float()
        FN = torch.sum((iou_values * targets_mask) < self.threshold).float()

        if self.det:
            return TP/self.det
        else:
            return TP/(TP+FN)
    
    def get_accuracy(self, predictions, targets):

        if len(targets.shape) > 2:
            n,c,_ = targets.shape 
            targets_mask = torch.ones((n,c))
        else: #Input shape of [N,4] is also acceptable
            n,_ = targets.shape 
            targets_mask = torch.ones(n)

        return self.get_recall(predictions, targets, targets_mask)

