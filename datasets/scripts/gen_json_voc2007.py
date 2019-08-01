#Compile all VOC XML annotation files into a single JSON file (one for each data split)
#Existing folder structure will remain the same
#Coordinates will be zero-indexed. Originally top-left pixel is (1,1)

import os
from glob import glob 
import xml.etree.ElementTree as ET
import json

source_root = '$DATA_DIRECTORY/VOC2007' #replace this value
target_root = '$JSON_DIRECTORY'         #replace this value


splits = ['train', 'val', 'test']

for split in splits:
    split_file = os.path.join(source_root,'ImageSets/Main/'+split+'.txt')
    target_file = os.path.join(target_root,split+'.json')

    print('Compiling annotations from: {}'.format(split_file))

    #Get all image names from split
    with open(split_file, 'r') as f:
        image_names = f.read().splitlines()

    ann_files = [] #xml annotation files
    for img in image_names:
        ann_files.append(os.path.join(source_root,'Annotations',img+'.xml'))

    json_ann = []

    #Parse through XML files and add to JSON dictionary
    base_path = os.path.join(source_root, 'JPEGImages')
    for f in ann_files:
        frames = []

        #An extra loop would be here if this was a video dataset
        frame = {}
        root = ET.parse(f).getroot()

        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)
        img_path = root.find('filename').text

        objects = root.findall('object')
        occluded = 0 #No occluded annotations on VOC_2007

        track_id = 0

        objs = []
        for obj in objects:
            class_name = obj.find('name').text  

            #zero-index coordinates
            xmin = int(obj.find('bndbox/xmin').text)-1
            ymin = int(obj.find('bndbox/ymin').text)-1
            xmax = int(obj.find('bndbox/xmax').text)-1
            ymax = int(obj.find('bndbox/ymax').text)-1

            truncated = int(obj.find('truncated').text)
            difficult = int(obj.find('difficult').text)

            objs.append({'trackid':track_id, 'c':class_name, 'occ':occluded, 'truncated':truncated, 'difficult':difficult, 'bbox':[xmin, ymin, xmax, ymax]})
            track_id += 1

        frame['objs'] = objs 
        frame['img_path'] =  img_path

        frames.append(frame)
        json_ann.append({'frames':frames, 'base_path':base_path, 'frame_size':[width, height]})

    #Write out to JSON file
    with open(target_file, 'w') as f:
        json.dump(json_ann, f)

