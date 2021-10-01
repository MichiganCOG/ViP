import json

#Recognition Dataset skeleton

action = {'action_class':'ACTION_CLASS'} 
frame   = {'frame_size (int, int)':'(WIDTH,HEIGHT)', 'img_path (str)':'FRAME_PATH',\
        'actions (list)': [action]}

frames = [frame]
vids = {'frames (list)':frames, 'base_path (str)':'BASE_VID_PATH'}
json_data = [vids]

with open('./datasets/templates/action_recognition_template.json', 'w') as f:
    json.dump(json_data, f, indent=4)


#Detection Dataset skeleton

obj = {'trackid (int)':'TRACK_ID', 'c (str or int)':'CLASS_ID', 'bbox (int,int,int,int)':'(xmin, ymin, xmax, ymax)', '(Optional) occ (int)':'Indicating if object is occluded in current frame'} 
frame   = {'frame_size (int, int)':'(WIDTH,HEIGHT)', 'img_path (str)':'FRAME_PATH',\
        'objs (list)': [obj]}

frames = [frame]
vids = {'frames (list)':frames, 'base_path (str)':'BASE_VID_PATH'}
json_data = [vids]

with open('./datasets/templates/detection_template.json', 'w') as f:
    json.dump(json_data, f, indent=4)
