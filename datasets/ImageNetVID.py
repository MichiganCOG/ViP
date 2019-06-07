import torch
from abstract_datasets import DetectionDataset 
from PIL import Image
import os
import numpy as np
#import cv2


class ImageNetVID(DetectionDataset):
    def __init__(self, *args, **kwargs):
        super(ImageNetVID, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        # TODO Update placeholder getitem

        # Load the frame images individually along with the annotation xml data, then combine to form the video
        images  = []
        annots  = []

        vidname = self.samples[idx]
        base_data_path = os.path.join(self.dataset_dir,'Data','VID',self.train_or_val, vidname)
        base_anno_path = os.path.join(self.dataset_dir,'Annotations','VID',self.train_or_val, vidname)

        frames = os.listdir(base_data_path)
        frames.sort()

        # Initialize the bbox coordinates and labels to the maximum number possible in the dataset (max is 22 objects in a single frame for imagenet vid)
        # Init the coordinates to -1
        xmin    = np.zeros((len(frames),self.maxbboxes))-1 
        xmax    = np.zeros((len(frames),self.maxbboxes))-1 
        ymin    = np.zeros((len(frames),self.maxbboxes))-1 
        ymax    = np.zeros((len(frames),self.maxbboxes))-1 
        label   = np.zeros((len(frames),self.maxbboxes))-1 

        frame_cnt = 0

        # Load each frame image and annotation
        for frame_id in frames:
            img = Image.open(os.path.join(base_data_path, frame_id))
            img_shape = img.shape
            img = Image.resize(img, (self.resize_shape[0], self.resize_shape[1]))

            # Image values range from 0 to 1
            img = img/255.
            images.append(img)

            annot_id = frame_id.split('.')[0]+'.xml'
            annot_f = open(os.path.join(base_anno_path, annot_id), 'r')
            annot = xmltodict.parse(annot_f)
            annot_f.close()
            annots.append(annot)

            # Check if there exist any bounding boxes in the frame
            if 'object' in annot['annotation'].keys():

                # Check if there exist more than one bounding box
                if type(annot['annotation']['object']) == list:
                    for bbox_cnt in range(len(annot['annotation']['object'])):
                        curr_xmax  = int(annot['annotation']['object'][bbox_cnt]['bndbox']['xmax'])
                        curr_xmin  = int(annot['annotation']['object'][bbox_cnt]['bndbox']['xmin'])
                        curr_ymax  = int(annot['annotation']['object'][bbox_cnt]['bndbox']['ymax'])
                        curr_ymin  = int(annot['annotation']['object'][bbox_cnt]['bndbox']['ymin'])

                        # Get the dataset label for the bbox object
                        temp_label = annot['annotation']['object'][bbox_cnt]['name']

                        # Get the human readable label for the bbox object (TODO: currently not used, figure out how to return string)
                        curr_label_name = self.labels_dict[temp_label] 

                        # Get the integer label for the bbox object
                        curr_label = int(self.label_values.index(curr_label_name))

                        # Resize bbox to match the resized image
                        curr_xmin, curr_xmax, curr_ymin, curr_ymax = self.resize_bbox(curr_xmin, curr_xmax, curr_ymin, curr_ymax, img_shape, self.resize_shape)
                                                                                                                                                                
                        
                        xmax[frame_cnt, bbox_cnt]  = curr_xmax
                        xmin[frame_cnt, bbox_cnt]  = curr_xmin
                        ymax[frame_cnt, bbox_cnt]  = curr_ymax
                        ymin[frame_cnt, bbox_cnt]  = curr_ymin
                        label[frame_cnt, bbox_cnt] = curr_label



                else:
                    curr_xmax = int(annot['annotation']['object']['bndbox']['xmax'])
                    curr_xmin = int(annot['annotation']['object']['bndbox']['xmin'])
                    curr_ymax = int(annot['annotation']['object']['bndbox']['ymax'])
                    curr_ymin = int(annot['annotation']['object']['bndbox']['ymin'])

                    # Get the dataset label for the bbox object
                    temp_label = annot['annotation']['object']['name']

                    # Get the human readable label for the bbox object (TODO: currently not used, figure out how to return string)
                    curr_label_name = self.labels_dict[temp_label] 

                    # Get the integer label for the bbox object
                    curr_label = int(self.label_values.index(curr_label_name))
            
                    # Resize bbox to match the resized image
                    curr_xmin, curr_xmax, curr_ymin, curr_ymax = self.resize_bbox(curr_xmin, curr_xmax, curr_ymin, curr_ymax, img_shape, self.resize_shape)
    
                    
                    xmax[frame_cnt, 0]  = curr_xmax
                    xmin[frame_cnt, 0]  = curr_xmin
                    ymax[frame_cnt, 0]  = curr_ymax
                    ymin[frame_cnt, 0]  = curr_ymin
                    label[frame_cnt, 0] = curr_label

            frame_cnt += 1

        
        # Extend the video and annotations if the length is lower than the required seq_length
        if len(images) < self.seq_length:
            images = np.repeat(images, math.ceil(self.seq_length/float(len(images))), axis=0).tolist()
            annots = np.repeat(annots, math.ceil(self.seq_length/float(len(annots))), axis=0).tolist()
            label  = np.repeat(label, math.ceil(self.seq_length/float(len(label))), axis=0).tolist()
            
            xmax = np.repeat(xmax, math.ceil(self.seq_length/float(len(xmax))), axis=0).tolist()
            xmin = np.repeat(xmin, math.ceil(self.seq_length/float(len(xmin))), axis=0).tolist()
            ymax = np.repeat(ymax, math.ceil(self.seq_length/float(len(ymax))), axis=0).tolist()
            ymin = np.repeat(ymin, math.ceil(self.seq_length/float(len(ymin))), axis=0).tolist()

        # Take the first seq_length frames and corresponding annotations
        images = images[:self.seq_length]
        annots = annots[:self.seq_length]
        label  = label[:self.seq_length]

        xmax = np.clip(xmax[:self.seq_length], a_min=-1, a_max=self.resize_shape[0]-1)
        xmin = np.clip(xmin[:self.seq_length], a_min=-1, a_max=self.resize_shape[0]-1)
        ymax = np.clip(ymax[:self.seq_length], a_min=-1, a_max=self.resize_shape[0]-1)
        ymin = np.clip(ymin[:self.seq_length], a_min=-1, a_max=self.resize_shape[0]-1)

        
        # Convert frames to pytorch tensor
        images = np.stack(images)
        images = torch.from_numpy(images)


        # Permute the opencv dimensions (Frame, Height, Width, Chan) to pytorch (Chan, frame, height, width) 
        images = images.permute(3, 0, 1, 2)


        xmax  = torch.from_numpy(np.array(xmax).astype(int))
        xmin  = torch.from_numpy(np.array(xmin).astype(int))
        ymax  = torch.from_numpy(np.array(ymax).astype(int))
        ymin  = torch.from_numpy(np.array(ymin).astype(int))
        label = torch.from_numpy(np.array(label).astype(int))


        ret_dict = {}
        ret_dict['data'] = images
        ret_dict['true_labels'] = label
        ret_dict['label_names'] = self.label_values
        #ret_dict['annots'] = annots
        ret_dict['xmax'] = xmax
        ret_dict['xmin'] = xmin
        ret_dict['ymax'] = ymax
        ret_dict['ymin'] = ymin

        return ret_dict


    def resize_bbox(self, xmin, xmax, ymin, ymax, img_shape, resize_shape):
        # Resize a bounding box within a frame relative to the amount that the frame was resized

        img_h = img_shape[0]
        img_w = img_shape[1]

        res_h = resize_shape[0]
        res_w = resize_shape[1]

        frac_h = res_h/float(img_h)
        frac_w = res_w/float(img_w)

        xmin_new = int(xmin * frac_w)
        xmax_new = int(xmax * frac_w)

        ymin_new = int(ymin * frac_h)
        ymax_new = int(ymax * frac_h)

        return xmin_new, xmax_new, ymin_new, ymax_new 

dataset = ImageNetVID(16, 128, 128)
import pdb; pdb.set_trace()
