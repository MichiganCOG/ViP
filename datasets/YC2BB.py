#Adapted from: https://github.com/MichiganCOG/Video-Grounding-from-Text
import torch
from .abstract_datasets import DetectionDataset 
from PIL import Image
import cv2
import os
import csv
import numpy as np

import torchtext

class YC2BB(DetectionDataset):
    '''
    YouCook2-Bounding Boxes dataset. Used in weakly-supervised video object grounding task
    Paper: https://arxiv.org/pdf/1805.02834.pdf
    '''
    def __init__(self, *args, **kwargs):
        super(YC2BB, self).__init__(*args, **kwargs)
        
        #Define the following configuration parameters in your config_*.yaml file
        #Or as a system arg
        class_file           = kwargs['yc2bb_class_file']
        num_proposals        = kwargs['yc2bb_num_proposals']
        rpn_proposal_root    = kwargs['yc2bb_rpn_proposal_root']
        roi_pooled_feat_root = kwargs['yc2bb_roi_pooled_feat_root']

        self.load_type = kwargs['load_type']

        self.max_objects = 20 
        self.class_dict  = _get_class_labels(class_file)
        '''
        if self.load_type=='train':
            self.transforms = kwargs['model_obj'].train_transforms
        else:
            self.transforms = kwargs['model_obj'].test_transforms
        '''

        sentences_proc, segments_tuple = _get_segments_and_sentences(self.samples, self.load_type)

        assert(len(sentences_proc) == len(segments_tuple))

        #YC2 split names, slightly different
        split_to_split = {'train':'training','val':'validation','test':'testing'}
        self.yc2_split = split_to_split[self.load_type]

	# read rpn object proposals
        self.rpn_dict = {}
        self.rpn_chunk = []

        total_num_proposals = 100 # always load all the proposals we have
        rpn_lst_file = os.path.join(rpn_proposal_root, self.yc2_split+'-box-'+str(total_num_proposals)+'.txt')
        rpn_chunk_file = os.path.join(rpn_proposal_root, self.yc2_split+'-box-'+str(total_num_proposals)+'.pth')
        key_counter = len(self.rpn_dict)
        with open(rpn_lst_file) as f:
            rpn_lst = f.readline().split(',')
            self.rpn_dict.update({r.strip():(i+key_counter) for i,r in enumerate(rpn_lst)})

        self.rpn_chunk.append(torch.load(rpn_chunk_file))

        self.rpn_chunk = torch.cat(self.rpn_chunk).cpu()
        assert(self.rpn_chunk.size(0) == len(self.rpn_dict))
        assert(self.rpn_chunk.size(2) == 4)

        self.num_proposals = num_proposals
        self.roi_pooled_feat_root = roi_pooled_feat_root

        '''
        with open(self.gt_box_file, 'r') as f:
            self.data_all = json.load(f)

	 # read gt bounding boxes O x T/25 x (id, ytl, xtl, ybr, xbr)
        # coordinates are 0-indexed
        for i, t in enumerate(segments_tuple):
            vid = t[2]
            seg = str(int(t[3]))

            # if video has no annotations, continue
            if not vid in self.data_all['database']:
                continue

            # check if ground truth bounding box exists for segment
            if seg in self.data_all['database'][vid]['segments'].keys():
                s = sentences_proc[i]
                inc_flag = 0
                obj_label = []
                for w in s:
                    if self.class_dict.get(w, -1) >= 0:
                        obj_label.append(self.class_dict[w])
                        inc_flag = 1

                if inc_flag:
                    self.sample_lst.append((t, obj_label))

        print('# of segments for {}: {}, percentage in the raw data: {:.2f}'.format(
               split_lst, len(self.sample_lst), len(self.sample_lst)/len(sentences_proc)))
        '''

    #Reverse-mapping between class index to canonical label name
    def _get_class_labels_reverse(self):
        return {v:k for k,v in self.class_dict.items()}

    def __getitem__(self, idx):
        vid_info = self.samples[idx]
        
        base_path       = vid_info['base_path']
        width, height   = vid_info['frame_size']
        num_frames_1fps = len(vid_info['frames'])
        rec             = base_path.split('/')[-3]
        vid             = base_path.split('/')[-2]
        seg             = base_path.split('/')[-1]

        bbox_data   = np.zeros((self.max_objects, num_frames_1fps, 5))-1 #[cls_label, xmin, ymin, xmax ymax]
        labels      = np.zeros(self.max_objects)-1

        for frame_ind in range(num_frames_1fps):
            frame      = vid_info['frames'][frame_ind]
            #frame_path = frame['img_path']
            num_objs    = len(frame['objs'])
            obj_label   = np.zeros((num_objs))-1 #List all unique class ids in entire segment
            
            # Extract bbox and label data from video info
            for obj_ind, obj in enumerate(frame['objs']):
                label   = self.class_dict[obj['c']]
                trackid = obj['trackid']

                if self.load_type == 'test' or self.load_type == 'train': #Annotations for test set not publicly available, train not annotated
                    bbox_data[trackid, frame_ind] = [label, -1, -1, -1, -1] 
                else:
                    if obj['occ'] or obj['outside']:
                        bbox_data[trackid, frame_ind] = [label, -1, -1, -1, -1] 
                    else:   
                        obj_bbox  = obj['bbox'] # [xmin, ymin, xmax, ymax]

                        #re-order to [ymin, xmin, ymax, xmax], rpn proposals are this way I believe
                        new_order = [1,0,3,2]
                        obj_bbox  = [obj_bbox[i] for i in new_order]
                        bbox_data[trackid, frame_ind, :] = [label] + obj_bbox

                obj_label[obj_ind] = label
                labels[trackid]    = label 

        #Only keep annotations for valid objects
        bbox_data = bbox_data[:num_objs, :]
        labels    = labels[:num_objs]

        obj_label = torch.from_numpy(obj_label).long()
        num_frames = num_frames_1fps * 25 #video sampled at 25 fps
        
        '''
	if self.vis_output:
            image_path = os.path.join(self.image_root, split, rec, vid, seg)
            img_notrans = []
            for i in range(num_frames):
                img_notrans.append(self.spatial_transform_notrans(self.loader(os.path.join(image_path, '{:04d}.jpg'.format(i+1)))))
            img_notrans = torch.stack(img_notrans, dim=1) # 3, T, H, W
        else:
            # no need to load raw images
            img_notrans = torch.zeros(3, num_frames, 1, 1) # dummy
        '''	

	# rpn object propoals
        rpn = []
        x_rpn = []
        frm=1

        feat_name = vid+'_'+seg+'.pth'
        img_name = vid+'_'+seg+'_'+str(frm).zfill(4)+'.jpg'
        x_rpn = torch.load(os.path.join(self.roi_pooled_feat_root, self.yc2_split, feat_name))
        while self.rpn_dict.get(img_name, -1) > -1:
            ind = self.rpn_dict[img_name]
            rpn.append(self.rpn_chunk[ind])
            frm+=1
            img_name = vid+'_'+seg+'_'+str(frm).zfill(4)+'.jpg'

        rpn = torch.stack(rpn) # number of frames x number of proposals per frame x 4
        rpn = rpn[:, :self.num_proposals, :]

        x_rpn = x_rpn.permute(2,0,1).contiguous() # encoding size x number of frames x number of proposals
        x_rpn = x_rpn[:, :, :self.num_proposals]

        rpn_original = rpn-1 # convert to 1-indexed

        # normalize coordidates to 0-1
        # coordinates are 1-indexed:  (x_tl, y_tl, x_br, y_br)
        rpn[:, :, 0] = (rpn[:, :, 0]-0.5)/width
        rpn[:, :, 2] = (rpn[:, :, 2]-0.5)/width
        rpn[:, :, 1] = (rpn[:, :, 1]-0.5)/height
        rpn[:, :, 3] = (rpn[:, :, 3]-0.5)/height

        assert(torch.max(rpn) <= 1)

        vis_name = '_-_'.join((self.yc2_split, rec, vid, seg))

        ret_dict = dict()
        ret_dict['data']     = [x_rpn, obj_label, self.load_type] 

        annot_dict = dict()
        annot_dict['box']               = bbox_data 
        annot_dict['box_label']         = obj_label 
        annot_dict['rpn']               = rpn
        annot_dict['rpn_original']      = rpn_original 
        annot_dict['vis_name']          = vis_name
        annot_dict['class_labels_dict'] = self._get_class_labels_reverse()
        ret_dict['annots']         = annot_dict

        return ret_dict

def _get_segments_and_sentences(data, split):
    # build vocab and tokenized sentences
    text_proc = torchtext.data.Field(sequential=True, tokenize='spacy',
                                lower=True, batch_first=True)
    split_sentences = []
    split_segments = []

    for dat in data:
        rec    = dat['base_path'].split('/')[-3]
        vid    = dat['base_path'].split('/')[-2]
        seg    = dat['base_path'].split('/')[-1]
        frame = dat['frames'][0]
        segment_labels = []
        if 'sentence' in frame: # for now, training json file only contains full sentence
            segment_labels = frame['sentence']
        else:
            for obj in frame['objs']:
                segment_labels.append(obj['c'])
        split_sentences.append(segment_labels)
        split_segments.append((split, rec, vid, str(seg).zfill(2))) #tuple of id (split, vid, seg)

    sentences_proc = list(map(text_proc.preprocess, split_sentences)) # build vocab on train and val

    print('{} sentences in {} split'.format(len(sentences_proc), split))

    return sentences_proc, split_segments

def _get_class_labels(class_file):
    class_dict = {} # both singular form & plural form are associated with the same label
    with open(class_file) as f:
        cls = csv.reader(f, delimiter=',')
        for i, row in enumerate(cls):
            for r in range(1, len(row)):
                if row[r]:
                    class_dict[row[r]] = int(row[0])

    return class_dict

