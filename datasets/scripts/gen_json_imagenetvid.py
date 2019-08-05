from os.path import join, isdir
from os import listdir
import os
import cv2
import argparse
import json
import glob
import xml.etree.ElementTree as ET


# Download the dataset from here: http://bvisionweb1.cs.unc.edu/ilsvrc2015/download-videos-3j16.php

label_mappings = {"n02374451": "horse", "n02691156": "airplane", "n02062744": "whale", "n01503061": "bird", "n03790512": "motorcycle ", "n02402425": "cattle", "n02342885": "hamster", "n04530566": "watercraft ", "n02958343": "car", "n02510455": "giant panda", "n02129165": "lion", "n02503517": "elephant", "n02129604": "tiger", "n02419796": "antelope", "n02391049": "zebra", "n02131653": "bear ", "n01674464": "lizard", "n04468005": "train", "n02509815": "red panda", "n02834778": "bicycle", "n02484322": "monkey", "n01726692": "snake", "n02084071": "dog", "n02324045": "rabbit", "n02924116": "bus", "n02118333": "fox", "n02355227": "squirrel", "n01662784": "turtle", "n02121808": "domestic cat", "n02411705": "sheep"}

def gen_label_keys():
    json.dump(label_mappings, open('labels_number_keys.json', 'w'))

def gen_json(load_type):
    VID_base_path = '/path/to/datasets/ILSVRC2015/' ###### REPLACE with the path to ImageNetVID
    ann_base_path = join(VID_base_path, 'Annotations/VID', load_type)
    img_base_path = join(VID_base_path, 'Data/VID', load_type)
    
    vid = []
    count = 0
    if load_type == 'train':
        vid_list = listdir(ann_base_path)
    else:
        vid_list = [''] 
    
    for curr_path in vid_list:
        if curr_path != '':
            curr_ann_path = join(ann_base_path, curr_path)
            curr_img_path = join(img_base_path, curr_path)
    
        else:
            curr_ann_path = ann_base_path
            curr_img_path = img_base_path
    
        videos = sorted(listdir(curr_img_path))
        for vi, video in enumerate(videos):
            v = dict()
            v['base_path'] = join(curr_img_path, video)
            v['frames'] = []
            video_base_path = join(curr_ann_path, video)
            if os.path.exists(video_base_path):
                xmls = sorted(glob.glob(join(video_base_path, '*.xml')))
                for xml in xmls:
                    f = dict()
                    xmltree = ET.parse(xml)
                    size = xmltree.findall('size')[0]
                    frame_sz = [int(it.text) for it in size]
                    objects = xmltree.findall('object')
                    objs = []
                    for object_iter in objects:
                        trackid = int(object_iter.find('trackid').text)
                        name = (object_iter.find('name')).text
                        bndbox = object_iter.find('bndbox')
                        occluded = int(object_iter.find('occluded').text)
                        o = dict()
                        o['c'] = name
                        o['bbox'] = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                                     int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]
                        o['trackid'] = trackid
                        o['occ'] = occluded
                        objs.append(o)
                    f['img_path'] = xml.split('/')[-1].replace('xml', 'JPEG')
                    f['objs'] = objs
                    v['frames'].append(f)
                    v['frame_size'] = frame_sz
            else:
                for img in sorted(listdir(join(curr_img_path, video))):
                    f = dict()
                    f['img_path'] = img 
                    f['objs'] = []
                    v['frames'].append(f)
                size = cv2.imread(join(curr_img_path, video, img)).shape
                v['frame_size'] = size[:2][::-1] 

            vid.append(v)
            count += 1
    
    print('save json (raw vid info), please wait 1 min~')
    json.dump(vid, open('ilsvrc_'+load_type+'.json', 'w'), indent=2)
    print('done!')

gen_json('test')
gen_json('val')
gen_json('train')
gen_label_keys()

