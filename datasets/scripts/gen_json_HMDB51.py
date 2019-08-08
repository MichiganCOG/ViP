# Generating JSON file compatible with ViP for HMDB51 dataset

import os
import sys
import json


splits      = ['train', 'test']
source_root = '$DATA_DIRECTORY/HMDB51'
target_root = '$JSON_DIRECTORY'

label_dict = {'brush_hair': 0,      'cartwheel': 1,
              'catch': 2,           'chew': 3,
              'clap': 4,            'climb_stairs': 5,
              'climb': 6,           'dive': 7,
              'draw_sword': 8,      'dribble': 9,
              'drink': 10,          'eat': 11,
              'fall_floor': 12,     'fencing': 13,
              'flic_flac': 14,      'golf': 15,
              'handstand': 16,      'hit': 17,
              'hug': 18,            'jump': 19,
              'kick_ball': 20,      'kick': 21,
              'kiss': 22,           'laugh': 23,
              'pick': 24,           'pour': 25,
              'pullup': 26,         'punch': 27,
              'push': 28,           'pushup': 29,
              'ride_bike': 30,      'ride_horse': 31,
              'run': 32,            'shake_hands': 33,
              'shoot_ball': 34,     'shoot_bow': 35,
              'shoot_gun': 36,      'sit': 37, 
              'situp': 38,          'smile': 39, 
              'smoke': 40,          'somersault': 41, 
              'stand': 42,          'swing_baseball': 43, 
              'sword_exercise': 44, 'sword': 45, 
              'talk': 46,           'throw': 47, 
              'turn': 48,           'walk': 49, 
              'wave': 50} 

for split in splits:
    base_path = os.path.join(source_root, split+'images')
    actions   = os.listdir(base_path) 

    dest_path = os.path.join(target_root, split+'.json')
    dest_data = [] 
    samples   = []
    count     = 0

    for action in actions:
        for video in os.listdir(os.path.join(base_path, action)):
            if not '.DS' in video:
                video_images = sorted(os.listdir(os.path.join(base_path, action, video.replace('.avi',''))))
                samples      = [os.path.join(base_path, action, video.replace('.avi',''), video_image) for video_image in video_images]
                    
                dest_data_lvl1           = {}
                dest_data_lvl1['frames'] = [] 
    
                for frame in samples:
                    dest_data_lvl1['frames'].append({'img_path': os.path.split(frame)[1],
                                                     'actions':[{'action_class': label_dict[action]}] })
    
                # END FOR
               
                dest_data_lvl1['base_path'] = os.path.split(frame)[0]
                dest_data.append(dest_data_lvl1)     
    
            # END IF
    
        # END FOR
    
    # END FOR
    
    with open(dest_path, 'w') as outfile:
        json.dump(dest_data, outfile, indent=4)
