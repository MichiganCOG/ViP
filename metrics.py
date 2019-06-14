import torch

class Metrics():
    def __init__(self, metric_type, *args, **kwargs):
        """
        Compute accuracy metrics from this Metrics class

        Args:
            metric_type: 

        """
        self.metric_type = metric_type 

        self.metric_object = None

        if self.metric_type == 'IOU':
            self.metric_object = IOU(*args, **kwargs)
        elif self.metric_type == 'mAP':
            self.metric_object = MAP(*args, **kwargs)
        elif self.metric_type =='AP':
            self.metric_object = AveragePrecision(*args, **kwargs)
        elif self.metric_type == 'AR':
            self.metric_object = AverageRecall(*args, **kwargs)

    def get_accuracy(self, predictions, targets):
        """
        Return accuracy from selected metric type

        Args:
            predictions: model predictions 
            targets: ground truth or targets 
        """

        return self.metric_object.get_accuracy(predictions, targets)

class IOU():
    """
    Intersection-over-union between two bounding boxes

    """

    def intersect(self, box_p, box_t):
        """
        Intersection area between both bounding boxes

        Args:
            box_p: prediction bounding box, shape [N,4]
            box_t: target bounding box, shape [N,4]

        Return:
            intersect area, shape [N,1]
        """

        assert torch.all(box_p[:,0] < box_p[:,2])
        assert torch.all(box_p[:,1] < box_p[:,3])
        assert torch.all(box_t[:,0] < box_t[:,2])
        assert torch.all(box_t[:,1] < box_t[:,3])

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

        box_p_area = (box_p[:,2] - box_p[:,0]) * (box_p[:,3] - box_p[:,1])
        box_t_area = (box_t[:,2] - box_t[:,0]) * (box_t[:,3] - box_t[:,1])
        union = box_p_area + box_t_area - intersect_area 
        
        return intersect_area / union 

    def get_accuracy(self, predictions, targets):
        """
        Args:
            predictions: shape [N,C,4], coordinate format [x1, y1, x2, y2]
            targets: shape [N,C,4]

        Return:
            iou: shape [N,C]
        """

        if len(predictions.shape) > 2:
            n,c,_ = predictions.shape
            iou_scores = torch.zeros((n,c))

            assert c == targets.shape[1]
            
            for cls in range(c):
                iou_scores[:,cls] = self.iou(predictions[:,cls,:], targets[:,cls,:])

            return iou_scores 
        else: #Input shape of [N,4] is also acceptable
            return self.iou(predictions, targets)

class AveragePrecision():
    """
    Compute Average Precision (AP)
    Args:
        threshold: (scalar) 
    """
    def __init__(self, threshold=0.5):

        self.threshold = threshold 
        self.IOU = IOU()

    def get_precision(self, predictions, targets, targets_mask):
        """
        Args:
            predictions: shape [N,C,4], coordinate format [x1, y1, x2, y2]
            targets: shape [N,C,4]
            targets_mask: binary mask, shape [N,C]
        """

        iou_values = self.IOU.get_accuracy(predictions, targets) #[N,C] 

        TP = torch.sum((iou_values * targets_mask) >= self.threshold).float()
        FP = torch.sum((iou_values * (1-targets_mask) >= self.threshold)).float()

        return TP/(TP+FP)
    
    #TODO: Complete accuracy in conjunction with Recall, get_precision uses threshold as input 
    def get_accuracy(self, predictions, targets):

        if len(targets.shape) > 2:
            n,c,_ = targets.shape 
            targets_mask = torch.ones((n,c))
        else: #Doesn't make sense for single class
            n,_ = targets.shape 
            targets_mask = torch.ones(n)

        return self.get_precision(predictions, targets, targets_mask) #[N,C]

class AverageRecall():
    """
    Compute Average Recall (AR)

    Args:
        threshold: (scalar)
        det: max number of detections per image (optional)
    """
    def __init__(self, threshold=0.5, det=None):
        
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

        return self.get_recall(predictions, targets, targets_mask) #[N,C]

class MAP():
    """
    Mean average precision

    Args:
    """
    def __init__(self, threshold=torch.linspace(0.5,0.95,10)):
        pass
