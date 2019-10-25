#Convert YC2-BB JSON annotation files to ViP JSON format

import os 
import json

source_root = '$ANNOTATIONS_ROOT/annotations' #replace this value
target_root = '$JSON_TARGET_ROOT' #replace this value
#Link to videos sampled at 1 fps
frame_root  = '$SAMPLED_FRAMES_ROOT' #replace this value
files = ['yc2_training_vid.json', 'yc2_bb_val_annotations.json', 'yc2_bb_public_test_annotations.json']

splits    = ['train', 'val', 'test']
ann_files = [os.path.join(source_root, f) for f in files]

for split, ann_file in zip(splits, ann_files):


    #YC2 split names, slightly different
    split_to_split = {'train':'training','val':'validation','test':'testing'}
    split_name = split_to_split[split]
    
    with open(ann_file) as f:
        ann_json_data = json.load(f)

    yc2_json_data = ann_json_data['database']
    json_data = []

    for vid_name in yc2_json_data.keys():
        frm_height  = yc2_json_data[vid_name]['rheight']
        frm_width   = yc2_json_data[vid_name]['rwidth']
        recipe_type = yc2_json_data[vid_name]['recipe_type'] 
        yc2_segments = yc2_json_data[vid_name]['segments']
        
        #Loop through segments, YC2 breaks down all each video into segment clips
        for seg,item in sorted(yc2_segments.items()):
            base_path   = os.path.join(frame_root, split_name, recipe_type, vid_name, str(seg).zfill(2))
            frames = []
            if 'objects' in item: #validation or testing file
                num_objs   = len(item['objects'])
                num_frames = len(item['objects'][0]['boxes'])

                #Loop through frames
                for f in range(num_frames):
                    frame = {}
                    objs = []

                    #Loop through objects
                    for track_id in range(num_objs):
                        obj = item['objects'][track_id]

                        cls_name = obj['label']
                        box    = obj['boxes'][f]
                        
                        if len(box) == 0: #No annotations
                            objs.append({'trackid':track_id, 'c':cls_name})
                            continue 

                        xmin = box['xtl']
                        ymin = box['ytl']
                        xmax = box['xbr']
                        ymax = box['ybr']

                        outside  = box['outside'] #outside or inside of frame
                        occluded = box['occluded'] 

                        objs.append({'trackid':track_id, 'c':cls_name, 'occ':occluded, 'outside':outside, 'bbox':[xmin, ymin, xmax, ymax]})

                    frame['img_path'] = os.path.join(base_path, str(seg).zfill(2), str(f).zfill(2)+'.jpg') 
                    frame['objs']     = objs 
                    frame['seg']      = seg
                    frames.append(frame) 
            else: #training annotation file
                frame = {}
                objs = []

                frame['sentence'] = yc2_segments[seg]['sentence'] 
                frame['objs']     = objs 
                frame['seg']      = seg
                frames.append(frame) 

            json_data.append({'frames':frames, 'base_path':base_path, 'frame_size':[frm_width, frm_height], 'recipe_type':recipe_type})

    target_file = os.path.join(target_root, split+'.json')
    print('Writing out to: {}'.format(target_file))
    with open(target_file, 'w') as f:
        json.dump(json_data, f)
        
