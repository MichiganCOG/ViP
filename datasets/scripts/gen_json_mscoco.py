import json
import os


year = '2014'

def save_json(load_type):
    
    # Define path to mscoco images data
    base_img_path = '/path/to/mscoco/images/'       ###### REPLACE with path to dataset
    base_annot_path = '/path/to/mscoco/annotations/'###### REPLACE with path to dataset

    save_location = '/path/to/save/location' ######### REPLACE with save path

    with open(os.path.join(base_annot_path,'instances_'+load_type+year+'.json'),'r') as f:
        x = json.load(f)
    
    imgids = [[idx['id'], idx['file_name'], idx['width'], idx['height']] for idx in x['images']]
    
    dd = {}
    for idx in imgids:
        frame_dict = dict(objs=[], img_path=idx[1]) 
        dd[idx[0]] = dict(frames=[frame_dict], base_path=os.path.join(base_img_path,load_type+year), frame_size=[idx[2],idx[3]])
    
    
    print('finished imgids')
    
    count = 0
    for annot in x['annotations']:
        image_id = annot['image_id']
        trackid = len(dd[image_id]['frames'][0]['objs'])  
        cat = annot['category_id']
        bbox = annot['bbox'] # [x,y,width,height]
        bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]] # [xmin, ymin, xmax, ymax]
        iscrowd=annot['iscrowd']
        obj_info = dict(trackid=trackid, c=cat, iscrowd=iscrowd, bbox=bbox)
        dd[image_id]['frames'][0]['objs'].append(obj_info)
        count+=1
        if count%1000==0:
            print(count)
    
    with open(os.path.join(save_location,load_type+'.json'), 'w') as f:
        json.dump(list(dd.values()), f)
                
 
save_json('train')
save_json('val')
