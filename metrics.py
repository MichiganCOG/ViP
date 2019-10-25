import os
import json 
import numpy as np

import torch

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
        elif self.metric_type == 'AveragePrecision':
            self.metric_object = AveragePrecision(*args, **kwargs)
        elif self.metric_type == 'mAP':
            self.metric_object = MAP(*args, **kwargs)
        elif self.metric_type == 'SSD_AP':
            self.metric_object = SSD_AP(*args, **kwargs)
        elif self.metric_type == 'Box_Accuracy':
            self.metric_object = Box_Accuracy(*args, **kwargs)
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

class Box_Accuracy():
    """
    Box accuracy computation for YC2-BB model.
    Adapted from: https://github.com/MichiganCOG/Video-Grounding-from-Text/blob/master/tools/test_util.py 

    Args:
        accu_thres: (float)  iou threshold
        fps:        (int)    frames per second video annotations were sampled at
        load_type:  (String) data split, only validation has publicly available annotations
        ndata       (int):   total number of datapoints in dataset 

    """
    def __init__(self, *args, **kwargs):
        from collections import defaultdict

        self.result_dir = os.path.join(kwargs['result_dir'], 'submission_yc2_bb.json')
        self.thresh     = kwargs['accu_thresh']
        self.fps        = kwargs['fps']
        self.debug      = kwargs['debug']
        self.test_mode  = 1 if kwargs['load_type'] == 'test' else 0
        self.IOU        = IOU()
        self.ba_score   = defaultdict(list) #box accuracy metric

        if self.test_mode:
            print('*'*62)
            print('* [WARNING] Eval unavailable for the test set! *\
                 \n* Results will be saved to: '+self.result_dir+' *\
                 \n* Please submit your results to the eval server!  *')
            print('*'*62)

        self.ndata = kwargs['ndata']
        self.count = 0
        
        self.json_data = {}
        self.database  = {}

    def get_accuracy(self, predictions, data):
        """
        Args:
            predictions: (Tensor, shape [N,W,T,D]), attention weight output from model
            data:      (dictionary)
                - rpn_original      (Tensor, shape [N,T,D,4]) 
                - box               (Tensor, shape [N,O,T,5]), [cls_label, ytl, xtl, ybr, xbr] (note order in coordinates is different) 
                - box_label         (Tensor, shape [N,W]) 
                - vis_name          (List, shape [N]), unique segment identifier  
                - class_labels_dict (dict, length 67) class index to class label mapping 

            T: number of frames
            D: dimension of features
            O: number of objects to ground 
            W: unique word in segment (from YC2BB class dictionary)
        Return:
           Box accuracy score  
        """
        attn_weights = predictions

        N = attn_weights.shape[0] 
        self.count += N

        rpn_batch         = data['rpn_original']
        box_batch         = data['box']
        obj_batch         = data['box_label']
        box_label_batch   = obj_batch 
        vis_name          = data['vis_name']
        class_labels_dict = data['class_labels_dict']

        # fps is the frame rate of the attention map
        # both rpn_batch and box_batch have fps=1
        _, T_rp, num_proposals, _ = rpn_batch.size()
        _, O, T_gt, _ = box_batch.size()
        T_attn = attn_weights.size(2)

        assert(T_rp == T_gt) # both sampled at 1fps
        #print('# of frames in gt: {}, # of frames in resampled attn. map: {}'.format(T_gt, np.rint(T_attn/self.fps)))

        hits, misses = [0 for o in range(O)], [0 for o in range(O)]

        results = []
        pos_counter = 0 
        neg_counter = 0 
        segment_dict = {} #segment dictionary - to output results to JSON file
        all_objects = []

        for o in range(O):
            object_dict = {}
            if box_label_batch[0, o] not in obj_batch[0, :]: 
                print('object {} is not grounded!'.format(box_label_batch[0, o]))
                continue # don't compute score if the object is not grounded
            obj_ind_in_attn = (obj_batch[0, :] == box_label_batch[0, o]).nonzero().squeeze()
            if obj_ind_in_attn.numel() > 1:
                obj_ind_in_attn = obj_ind_in_attn[0]
            else:
                obj_ind_in_attn = obj_ind_in_attn.item()

            new_attn_weights = attn_weights[0, obj_ind_in_attn]
            _, max_attn_ind = torch.max(new_attn_weights, dim=1)

            # uncomment this for the random baseline
            # max_attn_ind = torch.floor(torch.rand(T_attn)*num_proposals).long()
            label = class_labels_dict[box_label_batch[0,o].item()]
            object_dict = {'label':label}
        
            boxes = []
            for t in range(T_gt):
                if box_batch[0,o,t,0] == -1: # object is outside/non-exist/occlusion
                    boxes.append({'xtl':-1, 'ytl':-1, 'xbr':-1, 'ybr':-1, 'outside':1, 'occluded':1}) #object is either occluded or outside of frame 
                    neg_counter += 1
                    continue
                pos_counter += 1
                box_ind = max_attn_ind[int(min(np.rint(t*self.fps), T_attn-1))]
                box_coord = rpn_batch[0, t, box_ind, :].view(4) # x_tl, y_tl, x_br, y_br
                gt_box = box_batch[0,o,t][torch.Tensor([2,1,4,3]).type(box_batch.type()).long()].view(1,4) # inverse x and y

                if self.IOU.get_accuracy(box_coord, gt_box.float())[0].item() > self.thresh:
                    hits[o] += 1
                else:
                    misses[o] += 1

                xtl = box_coord[0].item()
                ytl = box_coord[1].item()
                xbr = box_coord[2].item()
                ybr = box_coord[3].item()
                boxes.append({'xtl':xtl, 'ytl':ytl, 'xbr':xbr, 'ybr':ybr, 'outside':0, 'occluded':0}) 

            object_dict['boxes'] = boxes
            all_objects.append(object_dict)

            results.append((box_label_batch[0, o].item(), hits[o], misses[o]))

        segment_dict['objects'] = all_objects
        #print('percentage of frames with box: {}'.format(pos_counter/(pos_counter+neg_counter)))
        
        for (i,h,m) in results:
            self.ba_score[i].append((h,m))

        #Annotations for the testing split are not publicly available
        if self.test_mode: 
            split, rec, video_name, segment = vis_name[0].split('_-_')

            if video_name not in self.database:
                self.database[video_name] = {}
                self.database[video_name]['recipe_type'] = rec
            if 'segments' not in self.database[video_name]:
                self.database[video_name]['segments'] = {}

            self.database[video_name]['segments'][int(segment)] = segment_dict 

            #Predictions will be saved to JSON file (if not in debug mode)
            if self.count >= self.ndata and not self.debug:
                self.json_data['database'] = self.database

                with open(self.result_dir, 'w') as f:
                    json.dump(self.json_data, f)

                print('Saved submission file to: {}'.format(self.result_dir))

            return -1

        ba_final = []
        for k, r in self.ba_score.items():
            cur_hit = 0 
            cur_miss = 0 
            for v in r:
                cur_hit += v[0]
                cur_miss += v[1]

            if cur_hit+cur_miss != 0:
                #print('BA for {}(...): {:.4f}'.format(k, cur_hit/(cur_hit+cur_miss)))
                ba_final.append(cur_hit/(cur_hit+cur_miss))

        return np.mean(ba_final)
