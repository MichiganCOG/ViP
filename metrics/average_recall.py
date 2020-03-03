import torch

class AverageRecall():
    #TODO: Incomplete
    def __init__(self, threshold=0.5, det=None, *args, **kwargs):
        """
        Compute Average Recall (AR)

        Args:
            threshold: (float)
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
