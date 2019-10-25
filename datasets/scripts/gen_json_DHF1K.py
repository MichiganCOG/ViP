import os
import cv2
import json


def get_split(base_vid_path):
    vids = os.listdir(base_vid_path)
    vids = [int(vid) for vid in vids]
    vids.sort()

    # Out of the 1000 videos, the first 600 are annotated for training, 601-700 annotated for val, 701-1000 not annotated must be sent in to test
    train_cutoff = 600
    val_cutoff = 700
    train_vids = vids[:vids.index(600)+1] 
    val_vids = vids[vids.index(600)+1:vids.index(700)+1] 
    test_vids = vids[vids.index(700)+1:]
    
    train_vids = [str(vid).zfill(3) for vid in train_vids]
    test_vids  = [str(vid).zfill(3) for vid in test_vids]
    val_vids   = [str(vid).zfill(3) for vid in val_vids]
    annot_train_vids = [vid.zfill(4) for vid in train_vids]
    annot_val_vids = [vid.zfill(4) for vid in val_vids]
    return train_vids, test_vids, val_vids, annot_train_vids, annot_val_vids


def save_json(load_type):
    base_vid_path = '/path/to/DHF1K/video_png'
    base_annot_path = '/path/to/DHF1K/annotation'
    output_path = '/any/path/'
   
    train_vids, test_vids, val_vids, annot_train, annot_val = get_split(base_vid_path)
    
    if load_type == 'train':
        tv_vids = train_vids
        tv_ann = annot_train
    elif load_type == 'val':
        tv_vids = val_vids
        tv_ann = annot_val

    else:
        tv_vids = test_vids
        tv_ann = []

    json_dat = [] 
    for vid in sorted(tv_vids):
        vid_dict = {}
        frames = []
        frame_size = []
        for img in sorted(os.listdir(os.path.join(base_vid_path, vid))):
            if frame_size == []:
                frame_shape = cv2.imread(os.path.join(base_vid_path, vid, img)).shape
                frame_size = [frame_shape[1], frame_shape[0]] # Width, Height
            frame_dict = {}
            frame_dict['img_path'] = img
            if load_type != 'test':
                frame_dict['map_path'] = os.path.join(base_annot_path, tv_ann[tv_vids.index(vid)], 'maps', img)
                frame_dict['bin_path'] = os.path.join(base_annot_path, tv_ann[tv_vids.index(vid)], 'fixation', img)
            else:
                frame_dict['map_path'] = '' 
                frame_dict['bin_path'] = ''

            frames.append(frame_dict)
        vid_dict['base_path'] = os.path.join(base_vid_path, vid)
        vid_dict['frames'] = frames
        vid_dict['frame_size'] = frame_size
        json_dat.append(vid_dict)

    writef = open(os.path.join(output_path,load_type+'.json'), 'w')
    json.dump(json_dat, writef)
    writef.close()

save_json('train')
save_json('val')
save_json('test')
