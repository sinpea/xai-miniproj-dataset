import json
from PIL import Image
from pathlib import Path

split_set='test'


if split_set=='train':
    index=1
    filename='cub_train_split1.json'
elif split_set=='test':
    index=0
    filename='cub_test_split1.json'


img_size=11788

data = {
    "info": {
        "description": "CUB-200-2011",
        "url": "https://data.caltech.edu/records/65de6-vp158",
        "version": "1.0",
        "year": 2022,
        "contributor": "Caltech Data",
        "date_created": " April 11, 2022"
    },
    "licenses": [
        {
            "id": 1,
            "name": "Wah, C., Branson, S., Welinder, P., Perona, P., & Belongie, S. (2022). CUB-200-2011 (1.0) CaltechDATA.",
            "url": "https://doi.org/10.22002/D1.20098"
        }
    ]
}

instance_ids = set()
with open('train_test_split.txt', 'r') as f:   #train: is_train==1, test: is_train==0
    for line in f:
        image_id, is_train = line.strip().split()
        if int(is_train) == index:
            instance_ids.add(int(image_id))


image_dict = []
with open('images.txt', 'r') as f:
    for line in f:
        image_id, filepath = line.strip().split()
        image_id=int(image_id)
        if image_id in instance_ids:
            imagepath=Path("images", filepath)
            image = Image.open(imagepath)  
            width, height = image.size
            image_dict.append({'license': 1, 'id': image_id, 'filename': filepath, 'width': width, 'height': height, 'background': 0})
data["images"]=image_dict


annotation_dict = {}
with open('bounding_boxes.txt', 'r') as f:
    for line in f:
        image_id, w, x, y, z = line.strip().split()
        image_id=int(image_id)
        if image_id in instance_ids:
            if image_id not in annotation_dict:
                annotation_dict[image_id] = []
            annotation_dict[image_id].append({'id': (22222 + image_id), 'image_id': image_id, 'bbox': [float(w), float(x), float(y), float(z)], "area": (float(y) * float(z)+ 0.00001)})

with open('image_class_labels.txt', 'r') as f:
    for line in f:
        image_id, category_id = line.strip().split()
        image_id=int(image_id)
        if image_id in instance_ids:
            if image_id not in annotation_dict:
                annotation_dict[image_id] = []
            annotation_dict[image_id].append({"category_id": int(category_id),  "iscrowd": 0})


img_keypoints = {}
img_keypoints_count = {}
with open('parts/part_locs.txt', 'r') as f: #considering part_id's are in sequence always
    for line in f:
        image_id, part_id, x_cord, y_cord, is_visible = line.strip().split()
        image_id=int(image_id)
        if image_id in instance_ids:
            if image_id not in img_keypoints:
                img_keypoints[image_id] = []
                img_keypoints_count[image_id] = 0
            img_keypoints[image_id].append(int(float(x_cord)))
            img_keypoints[image_id].append(int(float(y_cord)))
            if int(float(x_cord))!=0 and int(float(y_cord))!=0:
                img_keypoints[image_id].append(2)
                img_keypoints_count[image_id]+=1
            else:
                img_keypoints[image_id].append(0)


for img in range(img_size+1):
    image_id= int(img+1)
    if image_id in instance_ids:
        combined_dict = {}
        annotation_dict[image_id].append({"num_keypoints": img_keypoints_count[image_id], "keypoints":img_keypoints[image_id]})
        for dictnary in annotation_dict[image_id]:
            combined_dict.update(dictnary)
        annotation_dict[image_id]=combined_dict

annotations = list(annotation_dict.values())
data["annotations"]=annotations



body_parts=[]
with open('parts/parts.txt', 'r') as f:
    for line in f:
        if len(line.strip().split())==2:
            part_id, part_name = line.strip().split()
            body_parts.append(part_name)
        elif len(line.strip().split())==3:
            part_id, part_name1, part_name2  = line.strip().split()
            part_name=part_name1 + " " + part_name2
            body_parts.append(part_name)












import numpy as np
from collections import defaultdict


min_attr_avail_percentage=29

def process_file_to_2d_array(file_path):
    all_rows = []
    with open(file_path, 'r') as file:
        for line in file:
            values = list(map(float, line.split()))
            all_rows.append(values)  # Add the row to the list
    
    data_2d = np.array(all_rows)
    binary_array = np.where(data_2d > min_attr_avail_percentage, 1, 0)
    
    return binary_array

file_path = 'attributes/class_attribute_labels_continuous.txt'  
binary_attributes_2D_in_category = process_file_to_2d_array(file_path)


def process_attributes_file(file_path, attribute_list, excepts):
    attribute_dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ', 1)
            if len(parts) < 2:
                continue  
            
            attribute_id = int(parts[0]) - 1  
            attribute_info = parts[1]
            
            attribute_name, attribute_value = attribute_info.split("::", 1)
            
            if attribute_name in attribute_list:

                if attribute_value not in excepts and ('_' in attribute_value or '-' in attribute_value):
                    attribute_value = attribute_value.split('_')[0].split('-')[0]
                
                attr_name_value_dict = {attribute_name: attribute_value}
                
                if attribute_id in attribute_dict:
                    attribute_dict[attribute_id].append(attr_name_value_dict)
                else:
                    attribute_dict[attribute_id] = [attr_name_value_dict]
    
    return attribute_dict

file_path = 'attributes.txt'  
attribute_list=['has_bill_shape', 'has_wing_color', 'has_breast_pattern', 'has_back_color',
'has_tail_shape', 'has_breast_color', 'has_throat_color', 'has_eye_color',
'has_forehead_color', 'has_nape_color', 'has_belly_color', 'has_wing_shape',
'has_back_pattern', 'has_tail_pattern', 'has_belly_pattern', 'has_leg_color', 
'has_bill_color', 'has_crown_color', 'has_wing_pattern']
excepts = ['all-purpose', 'multi-colored']  

attribute_id_dict = process_attributes_file(file_path, attribute_list, excepts)





def extract_attributes(binary_array, attribute_dict):
    rows_attributes_dict = {} 
    
    for row_num, row in enumerate(binary_array):
        row_attributes = [] 
        
        for attribute_id, value in enumerate(row):
            if value == 1 and attribute_id in attribute_dict:
                row_attributes.append(attribute_dict[attribute_id])
        
        rows_attributes_dict[row_num] = row_attributes
    
    return rows_attributes_dict

attributes_by_row = extract_attributes(binary_attributes_2D_in_category, attribute_id_dict)

# print (attributes_by_row[0], '\n')



for key, attributes in attributes_by_row.items():
    combined_dict = defaultdict(list)
    
    for attr_list in attributes:
        for attr in attr_list:
            for attribute_key, attribute_value in attr.items():
                attribute_key = attribute_key.replace('bill', 'beak')
                combined_dict[attribute_key].append(attribute_value)
    
    for attribute_key in combined_dict:
        if len(combined_dict[attribute_key]) > 1:
            combined_dict[attribute_key] = tuple(combined_dict[attribute_key])
        else:
            combined_dict[attribute_key] = combined_dict[attribute_key][0] 
    
    attributes_by_row[key] = [dict(combined_dict)]



body_parts=[]
with open('parts/parts.txt', 'r') as f:
    for line in f:
        if len(line.strip().split())==2:
            part_id, part_name = line.strip().split()
            body_parts.append(part_name)
        elif len(line.strip().split())==3:
            part_id, part_name1, part_name2  = line.strip().split()
            part_name=part_name1 + " " + part_name2
            body_parts.append(part_name)




keypoint_attributes_by_category_template={part: [] for part in body_parts}


keypoint_attributes_by_category_results = {}

for key, attr_dict_list in attributes_by_row.items():
    keypoint_attributes_by_category = {k: [] for k in keypoint_attributes_by_category_template.keys()}

    for attr_dict in attr_dict_list:
        for full_key, value in attr_dict.items():
            key_parts = full_key.split('_')

            if len(key_parts) == 3:
                part, sub_part = key_parts[1], key_parts[2]

                if part == 'eye':
                    keypoint_attributes_by_category['right eye'].append({sub_part: value})
                    keypoint_attributes_by_category['left eye'].append({sub_part: value})
                elif part == 'leg':
                    keypoint_attributes_by_category['right leg'].append({sub_part: value})
                    keypoint_attributes_by_category['left leg'].append({sub_part: value})
                elif part == 'wing':
                    keypoint_attributes_by_category['right wing'].append({sub_part: value})
                    keypoint_attributes_by_category['left wing'].append({sub_part: value})
                else:
                    for keypoint, values_list in keypoint_attributes_by_category.items():
                        if keypoint.split()[-1] == part:
                            keypoint_attributes_by_category[keypoint].append({sub_part: value})

    keypoint_attributes_by_category_results[key] = keypoint_attributes_by_category









# keypoint_attributes_by_category_results


category_dict={}
with open('classes.txt', 'r') as f:
    for line in f:
        category, class_name = line.strip().split()
        category=int(category)
        class_name = class_name[4:].replace("_", " ")
        attributes=[]

        # print ("category", category)
        if category not in category_dict:
            category_dict[category] = []
        category_dict[category].append({'id':category, 'name': class_name, 'supercategory': 'bird', 'keypoints':body_parts, 'skeleton':[], 'keypoint_attributes_by_category': keypoint_attributes_by_category_results[category-1]})



categories = category_dict.values()
categories = [category[0] for category in categories]
data["categories"]=categories




json_string = json.dumps(data, separators=(",", ":"))
with open(filename, "w") as json_file:
    json_file.write(json_string)

print("JSON data has been written to your_file.json (one line)")


