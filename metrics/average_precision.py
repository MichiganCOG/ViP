import torch

class AveragePrecision():

    """
    Average Precision is computed per class and then averaged across all classes
    """

    def __init__(self, threshold=0.5, num_points=101, *args, **kwargs):
        """
        Compute Average Precision (AP)
        Args:
            threshold  (float): iou threshold 
            num_points (int): number of points to average for the interpolated AP calculation

        Return:
            None 
        """

        self.threshold = threshold 
        self.num_points = num_points
        self.IOU = IOU(average=False)

        self.result_dir = kwargs['result_dir']
        final_shape = kwargs['final_shape']
        #assuming model predictions are normalized between 0-1
        self.scale = torch.Tensor([1, final_shape[0], final_shape[1], final_shape[0], final_shape[1]]) #[1, height, width, height, width]

        self.ndata = kwargs['ndata']
        self.count = 0

    def update_threshold(self, threshold):
        self.threshold = threshold

    def compute_class_ap(self, tp, fp, npos):
        """
        Args:
            tp   (Tensor, shape [N*D]): cumulative sum of true positive detections 
            fp   (Tensor, shape [N*D]): cumulative sum of false positive detections 
            npos (Tensor, int): actual positives (from ground truth)

        Return:
            ap (Tensor, float): average precision calculation
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
    
    def get_AP(self, predictions, targets):
        """
        Args:
            predictions (Tensor, shape [N,C,D,5]): prediction bounding boxes, coordinate format [confidence, x1, y1, x2, y2]
            targets     (Tensor, shape [N,C,D_,4]): ground truth bounding boxes 
            C:  num of classes + 1 (0th class is background class, not included in calculation)
            D:  predicted detections
            D_: ground truth detections 

        Return:
            avg_ap (Tensor, float): mean ap across all classes 
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
            ap.append(self.compute_class_ap(tp, fp, npos)) #add class Average Precision

        #Average across all classes
        avg_ap = torch.mean(torch.Tensor(ap))
        return avg_ap

    def get_accuracy(self, detections, data):
        """
        Args:
            detections (Tensor, shape [N,C,D,5]): predicted detections, each item [confidence, x1, y1, x2, y2]
            data:      (dictionary)
                - labels      (Tensor, shape [N,T,D_,5]):, each item [x1, y1, x2, y3, class] 

        Return:
           Computes Average Precision  
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

        #Only compute Average Precision after accumulating all predictions
        if self.count < self.ndata:
            return -1

        self.targets = -1*torch.ones(self.ndata,C,D_,4)
        for n, trgt in enumerate(self._targets):
            for d_ in range(D_):
                c = trgt[d_,-1].long() + 1 #c=0 is now the background class

                if c != 0:
                    self.targets[n,c,d_] = trgt[d_,:4]

        return self.get_AP(self.predictions, self.targets)
