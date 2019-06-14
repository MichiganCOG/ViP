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
        elif self.metric_type == 'MAP':
            self.metric_object = MAP(*args, **kwargs)
        elif self.metric_type == 'Recall':
            self.metric_object = Recall(*args, **kwargs)

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

    Args:
        (None most likely)

    """
    def __init__(self):
        pass

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
            box_p: prediction bounding box, shape [N,4]
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
            predictions: shape [N,4], coordinate format [x1, y1, x2, y2]
            targets: shape [N,4]

        Return:
            iou: shape [N,1]
        """

        return self.iou(predictions, targets)

class MAP():
    """
    Mean average precision

    Args:
    """
    pass


class Recall():
    """
    Recall

    Args:
    """
    pass 

