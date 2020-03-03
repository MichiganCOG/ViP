import torch

from collections import defaultdict

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
