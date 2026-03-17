import re
import matplotlib.pyplot as plt

# get list elbow point
def get_elbow(labels,values,thresh=20):

    pairs = list(zip(labels,values))
    pairs.sort(key = lambda x : x[1],reverse=True)

    sorted_labels,sorted_list = zip(*pairs)

    sorted_labels = list(sorted_labels)
    sorted_list = list(sorted_list)

    # desc sorted list
    max_diff = 0
    max_loc = -1
    for idx in range(1,len(sorted_list)):
        if(sorted_list[idx-1] - sorted_list[idx] > max_diff):
            max_diff = sorted_list[idx-1] - sorted_list[idx]
            max_loc = idx

    # now we do some checks (threshold -- should be 29 according to dpsir code
    # but for our advantage we use 20 as noise floor

    while(sorted_list[max_loc] < thresh):
        max_loc -= 1
        if(max_loc == 0):
            break

    fin_values = sorted_list[:max_loc+1]
    fin_labels = sorted_labels[:max_loc+1]

    return fin_labels,fin_values

# fileattr = open("attributes.txt","r")
class_ex = int(input("Class no. (label - 1) to examine = "))


attribute_list = {}
attribute_index_ref = []
with open("attributes.txt",'r') as fileattr:
    previous = ""
    for line in fileattr:
        clean_line = line.strip()

        tokens = re.split(r'\shas_|::',clean_line)

        if(tokens[1] not in attribute_list):
            attribute_list[tokens[1]] = []
            
        attribute_index_ref.append((tokens[1],tokens[2]))
        attribute_list[tokens[1]].append(tokens[2])

# print([f'{idx} {k}' for idx,k in enumerate(attribute_list)])
# print(attribute_index_ref)
# at = input("choose attribute (string) = ")
class_names = []
with open ("./CUB_200_2011/classes.txt",'r') as classfile:
    for line in classfile:
        clean_line = line.strip()
        line_ext = re.split(r'\.',clean_line)
        print(line_ext)
        class_names.append(line_ext[-1])

# create the histogram data for each class
histo_data_cls = [{k : [] for k in attribute_list} for _ in range(200)]
#print(histo_data_cls[0])
with open("./CUB_200_2011/attributes/class_attribute_labels_continuous.txt",'r') as filespec:
    ln_no = 0
    for line in filespec:
        attrs = line.strip().split()
        #print(ln_no) 
        # get attribute selected
        #if ln_no == 0:
        #    print(histo_data_cls[ln_no])
        for idx,i in enumerate(attrs):
                
            histo_data_cls[ln_no][attribute_index_ref[idx][0]].append(float(i))
            if ln_no == 0:
                # print(attribute_index_ref[idx][0],attribute_index_ref[idx][1],histo_data_cls[ln_no][attribute_index_ref[idx][0]])
                pass

        ln_no += 1

# print(histo_data_cls[class_ex])
categories = []
for i in range(200):

    species_obj = {}
    max_thresh = 29
    sum_thresh = 65
    # now we try to obtain the values for each of the 28 fields an find the characteristics of bird
    for field in attribute_list:
        field_obj = get_elbow(attribute_list[field],histo_data_cls[i][field])
        species_obj[field] = (field_obj,(field_obj[1][0] > max_thresh and sum(field_obj[1]) > sum_thresh) or len(field_obj[1]) == 1)

    categories.append(species_obj)
    #print(species_obj)


def map_to_reference_format(extracted_data, species_id, species_name):
    # This dictionary maps your extracted field names to the 
    # Reference JSON keypoints and attribute names.
    mapping_logic = {
        "back": {"color": "back_color", "pattern": "back_pattern"},
        "beak": {"shape": "bill_shape", "color": "bill_color"},
        "belly": {"color": "belly_color", "pattern": "belly_pattern"},
        "breast": {"color": "breast_color", "pattern": "breast_pattern"},
        "crown": {"color": "crown_color"},
        "forehead": {"color": "forehead_color"},
        "left eye": {"color": "eye_color"},
        "right eye": {"color": "eye_color"},
        "left leg": {"color": "leg_color"},
        "right leg": {"color": "leg_color"},
        "left wing": {"color": "wing_color", "shape": "wing_shape", "pattern": "wing_pattern"},
        "right wing": {"color": "wing_color", "shape": "wing_shape", "pattern": "wing_pattern"},
        "nape": {"color": "nape_color"},
        "tail": {"pattern": "tail_pattern"},
        "throat": {"color": "throat_color"}
    }

    reference_json = {
        "id": species_id,
        "name": species_name,
        "supercategory": "bird",
        "keypoints": list(mapping_logic.keys()),
        "keypoint_attributes_by_category": {}
    }

    # Iterate through the hierarchy we defined above
    for keypoint, attributes in mapping_logic.items():
        keypoint_data = []
        
        for attr_name, source_field in attributes.items():
            if source_field in extracted_data:
                # Get the (([labels], [vals]), is_valid) structure
                (labels, values), is_valid = extracted_data[source_field]
                
                if True:
                    # Map the data. We use the labels list if it's multiple, 
                    # or a single string if it's just one.

                    attr_obj = {
                        attr_name:{
                            "value": labels if len(labels) > 1 else labels[0],
                            "confidence": [round(v, 2) for v in values],
                            "is_valid": is_valid
                        }
                    }
                    keypoint_data.append(attr_obj)
                    # val_to_insert = labels if len(labels) > 1 else labels[0]
                    # keypoint_data.append({attr_name: val_to_insert})
        
        # Only add the keypoint if we actually found valid data for it
        if keypoint_data:
            reference_json["keypoint_attributes_by_category"][keypoint] = keypoint_data

    return reference_json


categories_fmt = []
for idx,i in enumerate(class_names):
    formatted_output = map_to_reference_format(categories[idx], (idx+1), i)
    categories_fmt.append(formatted_output)

import json
# print(json.dumps(formatted_output, indent=4))
with open("catgories.json",'w') as f:
    json.dump(categories_fmt,f)
#fig,ax = plt.subplots(7,4,constrained_layout=True)
#ax = ax.flatten()

#for idx,i in enumerate(attribute_list):
#    ax[idx].bar(attribute_list[i],histo_data_cls[class_ex][i],color='skyblue', edgecolor='navy')



# plt.bar(attribute_list[at],histo_data_cls[class_ex][at],color='skyblue', edgecolor='navy')

# 2. Calculate the center of each bin
# (Bin 1 start + Bin 2 start) / 2

#plt.show()
"""
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

# 1. Create the Main Window
root = tk.Tk()
root.title("Scrollable Attributes")
root.geometry("800x600")

# 2. Add a Canvas and a Scrollbar
canvas = tk.Canvas(root)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

# Configure scrolling
scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

for idx,i in enumerate(attribute_list):
    plt.tight_layout()
    fig, ax = plt.subplots(figsize=(18, 10),constrained_layout=True)
    ax.bar(attribute_list[i],histo_data_cls[class_ex][i],color='skyblue', edgecolor='navy')
    ax.set_title(f"{i}")
    ax.tick_params(axis='x', rotation=45)
    
    # Optional: Align the text so the end of the word points to the bar
    plt.setp(ax.get_xticklabels(), ha="right") 
    # Embed the plot into the Tkinter frame
    chart_canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
    chart_canvas.draw()
    chart_canvas.get_tk_widget().pack(pady=10)

# 4. Pack the Scrollbar and Main Canvas
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

root.mainloop()
"""
