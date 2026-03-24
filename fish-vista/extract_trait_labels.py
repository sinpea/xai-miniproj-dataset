import torch
import open_clip
from PIL import Image
import json
import pandas as pd
from pathlib import Path
import os
import numpy as np
from transformers import AutoProcessor, AutoModel

# end goal -> create a specieswise trait list 
#
# I now need to get the data for each image
# lets get the paths out first

# replace with segmentation_test later
species_dict = {}
segmentation_df = pd.read_csv("segmentation_train.csv")


source_image_paths = segmentation_df['source_filename'].tolist()
mask_image_paths = segmentation_df['filename'].tolist()


for i in range(len(mask_image_paths)):
    mask_image_paths[i] = mask_image_paths[i][:-4] + ".png"

# after getting out the paths, lets look at getting the segmentation masks loaded as a tensor for each image
# from the segmentation masks we can multiply with corresponding image and get the intermediate image and then send
# it into the model to get softmaxxed values.. we can keep pushing it to a dataframe/json
# after which we can output that as a csv/json



segmentation_traits = {0: "Background", 1: "Head", 2: "Eye", 3: "Dorsal fin", 4: "Pectoral fin", 5: "Pelvic fin", 6: "Anal fin", 7: "Caudal fin", 8: "Adipose fin", 9: "Barbel"}
bioclip_trait_labels = {
    # 0: Background - Usually ignored for traits, but good to classify to filter out bad masks.
    0: [
        "A photo of a plain background",
        "A photo of an underwater environment",
        "A photo of a ruler or color scale",
        "A photo of a person's hand or holding tool",
        "A photo of a blurry or unrecognizable area" # Non-target
    ],
    
    # 1: Head - Focuses on mouth position and general head shape.
    1: [
        "A photo of a fish head with a terminal (forward-pointing) mouth",
        "A photo of a fish head with a superior (upward-pointing) mouth",
        "A photo of a fish head with a subterminal (downward-pointing) mouth",
        "A photo of a blunt, rounded fish head",
        "A photo of an elongated, pointed fish head",
        "A photo of a blurry, damaged, or unrecognizable fish head" # Non-target
    ],
    
    # 2: Eye - Focuses on prominent features or relative size.
    2: [
        "A photo of a large fish eye",
        "A photo of a small fish eye",
        "A photo of a fish eye with a dark vertical bar or teardrop",
        "A photo of a fish eye with a bright reddish or orange iris",
        "A photo of a blurry, obscured, or unrecognizable eye" # Non-target
    ],
    
    # 3: Dorsal fin - Focuses on spines vs. rays and continuous vs. split shapes.
    3: [
        "A photo of a spiny dorsal fin with sharp rays",
        "A photo of a soft-rayed, flexible dorsal fin",
        "A photo of a long, continuous dorsal fin",
        "A photo of two distinctly separate dorsal fins",
        "A photo of a sail-like, highly elevated dorsal fin",
        "A photo of a folded, damaged, or unrecognizable dorsal fin" # Non-target
    ],
    
    # 4: Pectoral fin - Focuses on length and shape.
    4: [
        "A photo of a long, pointed, wing-like pectoral fin",
        "A photo of a short, rounded pectoral fin",
        "A photo of a broad, fan-like pectoral fin",
        "A photo of a folded, damaged, or unrecognizable pectoral fin" # Non-target
    ],
    
    # 5: Pelvic fin - Focuses on placement (thoracic vs. abdominal) and modifications.
    5: [
        "A photo of thoracic pelvic fins located forward near the gills",
        "A photo of abdominal pelvic fins located midway down the belly",
        "A photo of modified, long thread-like pelvic fins",
        "A photo of a folded, damaged, or unrecognizable pelvic fin" # Non-target
    ],
    
    # 6: Anal fin - Focuses on base length and overall shape.
    6: [
        "A photo of a short-based anal fin",
        "A photo of a long-based anal fin extending along the belly",
        "A photo of an anal fin with prominent, sharp front spines",
        "A photo of a rounded, fleshy anal fin",
        "A photo of a folded, damaged, or unrecognizable anal fin" # Non-target
    ],
    
    # 7: Caudal fin (Tail) - The most critical morphological identifier.
    7: [
        "A photo of a deeply forked caudal fin",
        "A photo of a truncate or square-edged caudal fin",
        "A photo of a rounded caudal fin",
        "A photo of an emarginate or slightly notched caudal fin",
        "A photo of a lunate or crescent-shaped caudal fin",
        "A photo of a pointed caudal fin",
        "A photo of a folded, damaged, or cut-off caudal fin" # Non-target
    ],
    
    # 8: Adipose fin - Focuses on presence/absence (often a binary trait).
    8: [
        "A photo of a small, fleshy adipose fin without rays",
        "A photo of a prominent, large adipose fin",
        "A photo of a fish back with no adipose fin visible", # Non-target / Null presence
        "A photo of a blurry or unrecognizable area" # Non-target
    ],
    
    # 9: Barbel - Focuses on presence, length, and type.
    9: [
        "A photo of long, thread-like barbels near the mouth",
        "A photo of short, thick, fleshy barbels",
        "A photo of multiple barbels surrounding the mouth",
        "A photo of a fish mouth with no visible barbels", # Non-target / Null presence
        "A photo of a blurry or unrecognizable object" # Non-target
    ]
}

seg_mask_path = "segmentation_masks/images"
img_path = "Images/"
processed_img_path = "crops/"


img_dict = []
annotations_dict = []

# check if the paths exist.
#for idx in range(len(mask_image_paths)):
for idx in range(10):
    # not all images will be found
    # i
    path_img = Path(img_path) / source_image_paths[idx]
    mask_img = Path(seg_mask_path) / mask_image_paths[idx]
    
    if Path.exists(path_img) and Path.exists(mask_img):
        # ok, we take the image path and get our individual part by part masked images
        img_mask = Image.open(os.path.join(seg_mask_path, mask_image_paths[idx]))
        img_mask_arr = np.asarray(img_mask)
        original_img = Image.open(os.path.join(img_path, source_image_paths[idx]))
        original_arr = np.asarray(original_img)
        
        
        # The Blacking of Image
        # we will later loop for each trait
        for trait_value in range(1,10):
            """
            # 1. Create a True/False mask for your specific trait
            binary_mask = (img_mask_arr == trait_value)

            # 2. Create a blank (black) image with the same shape as the original
            extracted_feature_img = np.zeros_like(original_arr)

            # 3. Copy only the pixels where the binary_mask is True
            # Note: If your original image is RGB, you might need to handle the 3 channels.
            if len(original_arr.shape) == 3: # It's an RGB image
                # Apply the 2D mask to all 3 color channels
                extracted_feature_img[binary_mask] = original_arr[binary_mask]
            else: # It's a grayscale image
                extracted_feature_img[binary_mask] = original_arr[binary_mask]
            """
            
            # The Cropping of image
            # 1. Create the binary mask
            binary_mask = (img_mask_arr == trait_value)

            # 2. Find the row (y) and column (x) coordinates where the mask is True
            coords = np.column_stack(np.where(binary_mask))

            if coords.size > 0:
                # 3. Find the minimum and maximum coordinates to create a bounding box
                margin_ratio = 0.25
                    
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                
                img_height, img_width = original_arr.shape[:2]
                # 4. Calculate the current dimensions of the tight box
                box_width = x_max - x_min
                box_height = y_max - y_min
                
                # 5. Calculate how many pixels to add to each side (e.g., 25%)
                pad_x = int(box_width * margin_ratio)
                pad_y = int(box_height * margin_ratio)
                
                # 6. Add the margin, using max() and min() to prevent out-of-bounds errors!
                final_x_min = max(0, x_min - pad_x)
                final_y_min = max(0, y_min - pad_y)
                final_x_max = min(img_width, x_max + pad_x)
                final_y_max = min(img_height, y_max + pad_y)
                
                
                # 4. Slice the original array to crop it
                # cropped_feature = extracted_feature_img[y_min:y_max+1, x_min:x_max+1]
                # cropped_feature = original_arr[y_min:y_max+1, x_min:x_max+1]
                cropped_feature = original_arr[final_y_min:final_y_max, final_x_min:final_x_max]
                # To view or save it:
                # Image.fromarray(cropped_feature).show()

                #Put it in a new folder for crops
                cropped_img_to_save = Image.fromarray(cropped_feature)
                Path(processed_img_path + segmentation_traits[trait_value]).mkdir(parents=True,exist_ok=True)
                
                # Save the image
                cropped_img_to_save.save(processed_img_path + segmentation_traits[trait_value] + "/" + mask_image_paths[idx])
                print("Saved cropped feature successfully to : ",processed_img_path + mask_image_paths[idx])
                
                # Save the final bounding boxes and stuff to image data
            else:
                print("Trait not found in this mask.")
        
        
    else:
        print("Image not found")
# To view or save it:
# Image.fromarray(extracted_feature_img).show()
# after we are done with that, we need a trait based identification training set
# CUBS has a training set with bounding boxes with annotations for each part.
# we will do something similar, we will just put a path to the segmentation mask to that or do our best to create
# a bounding box using min_x,min_y,max_x,max_y approach of the mask
# 
# after we have that, we can create image wise Heirarchical dataset in COCO format

# 1. Setup Device

model_id = "google/siglip-base-patch16-224"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
model.eval() # Set to evaluation mode

# 2. Define your labels (Example for Mask ID 3: Dorsal Fin)
dorsal_fin_labels = [
    "A photo of a spiny dorsal fin with sharp rays",
    "A photo of a soft-rayed, flexible dorsal fin",
    "A photo of a long, continuous dorsal fin",
    "A photo of two distinctly separate dorsal fins",
    "A photo of a folded, damaged, or unrecognizable dorsal fin" # Non-target
]

# 3. PRE-COMPUTE TEXT EMBEDDINGS (The CPU Speed Secret)
# We do this outside the loop so the CPU only processes the text exactly once.
text_inputs = []
text_features = []

with torch.no_grad():
    for i in segmentation_traits:
        text_inputs.append(processor(text=bioclip_trait_labels[i], padding="max_length", return_tensors="pt"))
        text_features_i = model.get_text_features(**text_inputs[i])
        # Normalize the features for cosine similarity
        text_features.append(text_features_i.pooler_output / text_features_i.pooler_output.norm(p=2, dim=-1, keepdim=True))

# 4. Setup your dataset tracking
crops_path = Path("crops/")

# eh template
image_paths = [
    "crops/img_001_dorsal_fin.jpg", 
    "crops/img_002_dorsal_fin.jpg"
    # ... your actual list of cropped image paths
]
metadata_records = []

# 5. Process Images
print(f"Processing {len(image_paths)} crops on CPU...")

with torch.no_grad():
    for tr in range(1,10):
        if (crops_path/segmentation_traits[tr]).exists():
            image_paths = list((crops_path / segmentation_traits[tr]).iterdir())
            for img_path in image_paths:
                try:
                    # Load image
                    # print(img_path.name)
                    cmp_str = img_path.name[:-4] + ".jpg"
                    print(cmp_str)
                    species = segmentation_df.loc[segmentation_df['filename'] == cmp_str]['standardized_species'].iloc[0]
                    if(img_path.exists()):
                        image = Image.open(img_path).convert("RGB")
                    else:
                        height, width = 480, 640
                        # black image
                        image = Image.new('RGB', (width, height), color=(0,0,0))
                        
                    # The processor automatically resizes to 224x224 efficiently
                    image_inputs = processor(images=image, return_tensors="pt")
                    image_features_i = model.get_image_features(**image_inputs)
                    image_features = image_features_i.pooler_output / image_features_i.pooler_output.norm(p=2, dim=-1, keepdim=True)
                    
                    # Calculate Sigmoid probabilities
                    # Multiply image features by text features and apply SigLIP's temperature/bias
                    logits = (image_features @ text_features[tr].T) * model.logit_scale.exp() + model.logit_bias
                    probs = torch.sigmoid(logits).squeeze().tolist()
                    
                    # Find the highest scoring label
                    best_idx = probs.index(max(probs))
                    best_label = bioclip_trait_labels[tr][best_idx]
                    best_score = probs[best_idx]
                    # Record the data
                    metadata_records.append({
                        "image_path": img_path,
                        "predicted_trait": best_label,
                        "confidence_score": round(best_score, 4),
                        "is_valid": "unrecognizable" not in best_label # Flag non-targets instantly
                    })
                    
                    if species not in species_dict:
                        species_dict[species] = {}
                        species_dict[species][segmentation_traits[tr]] = [0] * len(bioclip_trait_labels[tr])
                        species_dict[species][segmentation_traits[tr]][best_idx] += 1 
                    else:
                        trait_name = segmentation_traits[tr]
                        if trait_name not in species_dict[species]:
                            species_dict[species][trait_name] = [0] * len(bioclip_trait_labels[tr])
                        species_dict[species][segmentation_traits[tr]][best_idx] += 1
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

# 6. Save to CSV
df = pd.DataFrame(metadata_records)
df.to_csv("dorsal_fin_traits.csv", index=False)

with open("species_wise.json","w") as f:
    json.dump(species_dict,f)

print("Processing complete! Saved to dorsal_fin_traits.csv")
