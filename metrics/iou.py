import torch

class IOU():
    """
    Intersection-over-union between one prediction bounding box 
    and plausible ground truth bounding boxes

    """
    def __init__(self, *args, **kwargs):
        pass    

    def intersect(self, box_p, box_t):
        """
        Intersection area between predicted bounding box and 
        all ground truth bounding boxes

        Args:
            box_p (Tensor, shape [4]): prediction bounding box, coordinate format [x1, y1, x2, y2]
            box_t (Tensor, shape [N,4]): target bounding boxes

        Return:
            intersect area (Tensor, shape [N]): intersect_area for all target bounding boxes
        """
        x_left = torch.max(box_p[0], box_t[:,0])
        y_top = torch.max(box_p[1], box_t[:,1])
        x_right = torch.min(box_p[2], box_t[:,2])
        y_bottom = torch.min(box_p[3], box_t[:,3])

        width = torch.clamp(x_right - x_left, min=0)
        height = torch.clamp(y_bottom - y_top, min=0)

        intersect_area = width * height

        return intersect_area

    def iou(self, box_p, box_t):
        """
        Performs intersection-over-union 

        Args:
            box_p (Tensor, shape [4]): prediction bounding box, coordinate format [x1, y1, x2, y2]
            box_t (Tensor, shape [N,4]): target bounding boxes

        Return:
            overlap (Tensor, shape [1]): max overlap
            ind     (Tensor, shape [1]): index of bounding box with largest overlap
        """

        intersect_area = self.intersect(box_p, box_t)

        box_p_area = (box_p[2] - box_p[0]) * (box_p[3] - box_p[1])
        box_t_area = (box_t[:,2] - box_t[:,0]) * (box_t[:,3] - box_t[:,1])
        union = box_p_area + box_t_area - intersect_area
        overlap = torch.max(intersect_area/union)
        ind     = torch.argmax(intersect_area/union)

        assert overlap >= 0.0
        assert overlap <= 1.0

        return overlap, ind

    def get_accuracy(self, prediction, targets):
        """
        Args:
            prediction (Tensor, shape [4]): prediction bounding box, coordinate format [x1, y1, x2, y2]
            targets    (Tensor, shape [N,4]): target bounding boxes

        Return:
            iou (Tensor, shape[1]): Highest iou amongst target bounding boxes
            ind (Tensor, shape[1]): Index of target bounding box with highest score
        """

        iou_score, ind = self.iou(prediction, targets)
        return iou_score, ind
