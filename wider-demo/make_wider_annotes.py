import os
import re
from tqdm import tqdm

import numpy as np
from PIL import Image
from scipy.io import loadmat


ROOTDIR  = '/Users/visenze/Datasets/wider'
ANNDIR   = os.path.join(ROOTDIR, 'annotations')
IMGDIR   = os.path.join(ROOTDIR, 'images')
DATASETS = ['train', 'test', 'val']


def parse_int_from_string(s):
    all_digits = ''.join(filter(str.isdigit, s))
    if all_digits == '':
        return None
    else:
        return int(all_digits)


def get_labels():
    return {
        'blur': {
            '0': {'id': 0, 'name': 'clear' , 'tag': 'blur'},
            '1': {'id': 1, 'name': 'normal', 'tag': 'blur'},
            '2': {'id': 2, 'name': 'heavy' , 'tag': 'blur'}
        },
        'expression': {
            '0': {'id': 0, 'name': 'typical'   , 'tag': 'expression'},
            '1': {'id': 1, 'name': 'exaggerate', 'tag': 'expression'}
        },
        'illumination': {
            '0': {'id': 0, 'name': 'normal' , 'tag': 'expression'},
            '1': {'id': 1, 'name': 'extreme', 'tag': 'expression'}
        },
        'occlusion': {
            '0': {'id': 0, 'name': 'none' , 'tag': 'illumination'},
            '1': {'id': 1, 'name': 'partial', 'tag': 'illumination'},
            '2': {'id': 2, 'name': 'heavy' , 'tag': 'illumination'}
        },
        'pose': {
            '0': {'id': 0, 'name': 'typical' , 'tag': 'pose'},
            '1': {'id': 1, 'name': 'atypical', 'tag': 'pose'}
        },
        'invalid': {
            '0': {'id': 0, 'name': 'false', 'tag': 'invalid'},
            '1': {'id': 1, 'name': 'true' , 'tag': 'invalid'}
        },
    }


def get_tags():
    return list(get_labels().keys())


def get_events(data):
    event_list = [str(event[0]) for event in data['event_list'][:, 0]]
    return event_list


def get_num_imgs(data):
    img_count = 0
    
    for event_id, event in enumerate(get_events(data)):
        file_list = [str(f[0][0]) for f in data['file_list'][event_id, 0]]
        img_count += len(file_list)
    
    return img_count
    

def get_img_infos(data):
    all_img_info = {}
    img_count = 0
    
    with tqdm(total=get_num_imgs(data)) as pbar:
    
        for event_id, event in enumerate(get_events(data)):
            file_list = [str(f[0][0]) for f in data['file_list'][event_id, 0]]

            for file_id, file_name in enumerate(file_list):
                # Get image info
                img_info = {
                    'id'       : img_count,
                    'event'    : event,
                    'file_path': os.path.join('images', event, file_name) + '.jpg'
                }

                img = Image.open(os.path.join(ROOTDIR, img_info['file_path']))
                img = np.array(img)

                img_info['height'] = img.shape[1]
                img_info['width']  = img.shape[0]

                # Store info
                all_img_info[img_count] = img_info

                # Update progress
                img_count += 1
                pbar.update(1)
    
    return all_img_info


def get_anns(data):
    all_img_to_anns = {}
    all_anns = {}
    img_count = 0
    ann_count = 0
    
    for event_id, event in enumerate(get_events(data)):
            file_list = [str(f[0][0]) for f in data['file_list'][event_id, 0]]

            for file_id, file_name in enumerate(file_list):
                anns_for_img = []
                
                for ann_idx, ann in enumerate(data['face_bbx_list'][event_id, 0][file_id, 0]):
                    # Get ann info
                    ann_info = {
                        'id'      : ann_count,
                        'image_id': img_count,
                        'area'    : ann[2] * ann[3],
                        'bbox'    : ann.tolist(),
                    }
                    
                    for tag in get_tags():
                        ann_info[tag] = data[tag + '_label_list'][event_id, 0][file_id, 0][ann_idx, 0]
                    
                    # Store info
                    anns_for_img.append(ann_info)
                    all_anns[ann_count] = ann_info
                    
                    # Update progress
                    ann_count += 1
                
                # Store info
                all_img_to_anns[img_count] = anns_for_img
                
                # Update progress
                img_count += 1
    
    return all_img_to_anns, all_anns


def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def main():
    for dataset_name in DATASETS:
        json_filepath = os.path.join(ANNDIR, 'wider_face_{}.json'.format(dataset_name))
        
        # Load data
        data = loadmat(os.path.join(ANNDIR, 'wider_face_{}.mat'.format(dataset_name)))        
        
        # Create template to store data later
        data_json = {}
        
        data_json['labels'] = get_labels()
        data_json['tags']   = get_tags()
        data_json['events'] = get_events(data)
        
        data_json['num_imgs']  = get_num_imgs(data)
        data_json['img_infos'] = get_img_infos(data)
        
        if dataset_name != 'test':
            data_json['img_to_Anns'], data_json['anns'] = get_anns(data)
        
        save_json(data_json, json_filepath)


if __name__ == '__main__':
    main()

