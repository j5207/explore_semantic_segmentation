
# coding: utf-8

import json
import matplotlib.pyplot as plt
import scipy.misc as m
import os, sys, time, re, json
import glob
import numpy as np
import imageio


with open ('rr_573_mapping.json') as f:
    json_mapping = json.load(f)



object_2_mask = {}
for key in json_mapping['object_mask_colors']:
    object_2_mask[key] = 'background'   


for key in json_mapping['object_mask_colors']:
    if 'TV' in key:
        object_2_mask[key] = 'tvmonitor'
    if 'Couch' in key:
        object_2_mask[key] = 'sofa'
    if 'Plant' in key:
        object_2_mask[key] = 'pottedplant'


valid_classes = [0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11, 12, 13, 14, 15, 16, 17, 18, 19,20]
class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',                    'bus', 'car', 'cat', 'chair', 'cow',                    'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant', 'sheep', 'sofa',                     'train', 'tvmonitor']
class_map = dict(zip(class_names,valid_classes)) 
print(class_map)



class Color(object):
    ''' A utility class to parse color value '''
    regexp = re.compile('\(R=(.*),G=(.*),B=(.*),A=(.*)\)')
    def __init__(self, color_str):
        self.color_str = color_str
        match = self.regexp.match(color_str)
        (self.R, self.G, self.B, self.A) = [int(match.group(i)) for i in range(1,5)]

    def __repr__(self):
        return self.color_str
def match_color(object_mask, target_color, tolerance=3):
    match_region = np.ones(object_mask.shape[0:2], dtype=bool)
    for c in range(3): # r,g,b
        min_val = target_color[c] - tolerance
        max_val = target_color[c] + tolerance
        channel_region = (object_mask[:,:,c] >= min_val) & (object_mask[:,:,c] <= max_val)
        match_region &= channel_region

    if match_region.sum() != 0:
        return match_region
    else:
        return None

object_masks = glob.glob('object_mask/*')
for mask_file in object_masks:
    ob_mask = imageio.imread(mask_file)
    ob_mask = np.array(ob_mask,dtype=np.uint8)
    output = np.zeros(ob_mask.shape[0:2])
    id2mask = {}
    for obj_id in json_mapping['object_mask_colors']:
        color = json_mapping['object_mask_colors'][obj_id]
        mask = match_color(ob_mask, [color['r'], color['g'], color['b']], tolerance = 10)
        if mask is not None:
            id2mask[obj_id] = mask
            output[mask] = class_map[object_2_mask[obj_id]]    
    output[output==0] = 250
    # Prevent dynamic scaling
    m.toimage(output,cmin=0,cmax=255).save('object_mask_remap'+mask_file[11:])
    #imageio.imwrite(object_masks[0][:-4]+ '_remap.png',output)

