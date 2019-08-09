import torch
import numpy as np

class Metrics(object):
    def __init__(self, *args, **kwargs):
        """
        Compute accuracy metrics from this Metrics class
        Args:
            acc_metric (String): String used to indicate selected accuracy metric 
    
        Return:
            None
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

        if self.metric_type == None:
            return -1

        else:
            return self.metric_object.get_accuracy(predictions, targets, **kwargs)

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

class Precision():

    def __init__(self, threshold=0.5, *args, **kwargs):

        self.threshold = threshold
        self.IOU = IOU(average=False)

    def get_precision(self, scores, targets_mask):
        """
        Args:
            scores       (Tensor, shape[N]): iou scores per prediction  
            targets_mask (Tensor, shape[N]): binary mask, positive class (1) negative class (0)

        Return:
            precision (Tensor, shape[1]): output precision
        """

        TP = torch.sum((scores * targets_mask) >= self.threshold).float()
        FP = torch.sum((scores * (1-targets_mask) >= self.threshold)).float()

        return TP/(TP+FP)

    def get_accuracy(self, predictions, targets):
        """
        Args:
            (assuming N predictions for N targets)
            predictions (Tensor, shape [N,4]): prediction bounding box, coordinate format [x1, y1, x2, y2]
            targets     (Tensor, shape [N,4]): target bounding boxes

        Return:
            precision   (Tensor, shape [1]): precision for predictions 
        """

        n,_ = targets.shape 
        targets_mask = torch.ones(n)
        scores = torch.zeros(n)

        for i in enumerate(scores):
            scores[i] = self.IOU.get_accuracy(predictions[i], targets[i].unsqueeze(0)) 
            if torch.equal(targets[i], torch.Tensor([-1,-1,-1,-1])):
                targets_mask[i] = 0

        return self.get_precision(scores, targets_mask)

class AveragePrecision():

    """
    Average Precision is computed per class and then averaged across all classes
    """

    def __init__(self, threshold=0.5, num_points=101, *args, **kwargs):
        """
        Compute Average Precision (AP)
        Args:
            threshold  (scalar): iou threshold 
            num_points (scalar): number of points to average for the interpolated AP calculation

        Return:
            None 
        """

        self.threshold = threshold 
        self.num_points = num_points
        self.IOU = IOU(average=False)

    def update_threshold(self, threshold):
        self.threshold = threshold

    def get_average_precision(self, tp, fp, npos):
        """
        Args:
            tp   (Tensor, shape [N*D]): cumulative sum of true positive detections 
            fp   (Tensor, shape [N*D]): cumulative sum of false positive detections 
            npos (Tensor, scalar): actual positives (from ground truth)

        Return:
            ap (Tensor, scalar): average precision calculation
        """
        
        #Values for precision-recall curve
        rc = tp/npos
        pr = tp / torch.clamp(tp + fp, min=torch.finfo(torch.float).eps)
        rc_values = torch.linspace(0,1,self.num_points) #sampled recall points for n-point precision-recall curve

        #The interpotaled P-R curve will take on the max precision value to the right at each recall
        ap = 0.
        for t in rc_values:
            if torch.sum(rc >= t) == 0:
                p = 0
            else:
                p = torch.max(pr[rc >= t])
            ap = ap + p/self.num_points

        return ap
    
    def get_accuracy(self, predictions, targets):
        """
        Args:
            predictions (Tensor, shape [N,C,D,5]): prediction bounding boxes, coordinate format [confidence, x1, y1, x2, y2]
            targets     (Tensor, shape [N,C,D_,4]): ground truth bounding boxes 
            C:  num of classes + 1 (0th class is background class, not included in calculation)
            D:  predicted detections
            D_: ground truth detections 

        Return:
            avg_ap (Tensor, scalar): mean ap across all classes 
        """

        N,C,D,_ = predictions.shape
        _,_,D_,_ = targets.shape 
        ap = []
        
        mask_g = torch.zeros(N,C,D_)
        for c in range(1,C): #skip background class (c=0)

            #Sort predictions in descending order, by confidence value
            pred = predictions[:,c].contiguous().view(N*D,-1)
            idx  = pred[:,0].argsort(descending=True)
            pred = pred[idx]

            img_labels = torch.arange(0,N).unsqueeze(1).repeat(1,D).view(N*D)
            img_labels = img_labels[idx]

            tp   = []
            fp   = []
            mask = torch.zeros(N,D_,dtype=torch.uint8)
            class_targets = targets[:,c]

            for i in range(class_targets.shape[0]):
                for j in range(class_targets.shape[1]):
                    if not torch.equal(class_targets[i,j], torch.Tensor([-1,-1,-1,-1])):
                        mask[i,j] = 1

            npos = torch.sum(mask)

            for n, p in zip(img_labels, pred[:,1:]): #get iou for all detections
                trgts = targets[n,c]

                gt_mask = mask[n]
                exists = torch.sum(gt_mask) > 0 #gt exists on this image
                if not torch.equal(p, torch.Tensor([0,0,0,0])):
                    if exists:
                        score, ind = self.IOU.get_accuracy(p,trgts[gt_mask])
                    else:
                        score = 0.0

                    if score > self.threshold:
                        if mask_g[n,c,ind] == 1: #duplicate detection (false positive)
                            tp.append(0.)
                            fp.append(1.)
                        else: #true positive
                            tp.append(1.)
                            fp.append(0.)
                            mask_g[n,c,ind] = 1
                    else: #below threshold (false positive)
                        tp.append(0.)
                        fp.append(1.)
                else:
                    break

            tp = torch.cumsum(torch.Tensor(tp), dim=0)
            fp = torch.cumsum(torch.Tensor(fp), dim=0)
            ap.append(self.get_average_precision(tp, fp, npos)) #add class Average Precision
            
        #Average across all classes
        avg_ap = torch.mean(torch.Tensor(ap))
        return avg_ap

class SSD_AP(AveragePrecision):
    """
    Compute Average Precision from the output of the SSD model
    Accumulates all predictions before computing AP
    """

    def __init__(self, threshold=0.5, num_points=11, *args, **kwargs):
        """
        Compute Average Precision (AP)
        Args:
            threshold    (scalar): iou threshold 
            num_points   (scalar): number of points to average for the interpolated AP calculation
            final_shape  (list) : [height, width] of input given to CNN
            result_dir   (String): save detections to this location
            ndata        (scalar): total number of datapoints in dataset 

        Return:
            None 
        """
        super(SSD_AP, self).__init__(threshold=threshold, num_points=num_points)

        self.result_dir = kwargs['result_dir']
        self.final_shape = kwargs['final_shape']

        self.ndata = kwargs['ndata']
        self.count = 0

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
        scale = torch.Tensor([1, self.final_shape[0], self.final_shape[1], self.final_shape[0], self.final_shape[1]]) #[1, height, width, height, width]

        detections = detections.data
        N,C,D,_    = detections.shape
        _,D_,_     = gt.shape 

        if self.count == 0:
            self.predictions = -1*torch.ones(self.ndata,C,D,5)
            self._targets    = -1*torch.ones(self.ndata,D_,5)
            self._diff       = torch.zeros(self.ndata,D_, dtype=torch.long)

        self.predictions[self.count:self.count+N] = detections * scale
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

        return super(SSD_AP,self).get_accuracy(self.predictions, self.targets)

class MAP():

    def __init__(self, threshold=torch.linspace(0.5,0.95,10), num_points=101, *args, **kwargs):
        """
        Mean average precision

        Args:
            threshold  (Tensor, shape[10]): Calculate AP at each of these threshold values
            num_points (scalar): number of points to average for the interpolated AP calculation
        """

        self.threshold = threshold
        self.IOU = IOU(average=False)
        self.AP = AveragePrecision(num_points=num_points)

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
            AP_scores[n] = self.AP.get_accuracy(predictions, targets)

        return torch.mean(AP_scores)

    def get_accuracy(self, predictions, targets):
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

        return self.get_mAP(predictions, targets)

class Recall():
    def __init__(self, threshold=0.5, *args, **kwargs):

        self.threshold = threshold
        self.IOU = IOU(average=False)

    def get_recall(self, scores, targets_mask):
        """
        Args:
            scores       (Tensor, shape[N]): iou scores per prediction 
            targets_mask (Tensor, shape[N]): binary mask, postive class (1) negative class (0)
        
        Return:
            recall (Tensor, shape[1]): output recall
        """
        TP = torch.sum((scores * targets_mask) >= self.threshold).float()
        FN = torch.sum((scores * targets_mask) < self.threshold).float()

        return TP/(TP+FN)

    def get_accuracy(self, predictions, targets):
        """
        Args:
            (assuming N predictions for targets)
            predictions (Tensor, shape [N,4]): prediction bounding box, coordinate format [x1, y1, x2, y2]
            targets     (Tensor, shape [N,4]): target bounding boxes
        
        Return:
            recall   (Tensor, shape [1]): recall for predictions 

        """
        n,c,_ = targets.shape 
        targets_mask = torch.ones((n,c))
        scores = self.IOU.get_accuracy(predictions, targets)

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

