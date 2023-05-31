import cv2 
import json
from segment import *
from alive_progress import alive_bar

# Obtain DICOM ids from REFLACX ids
with open("tmp/reflacx_map_id2meta.json") as reflacx_metadata:
    reflacx_map = json.load(reflacx_metadata)

# Store DICOM, REFLACX id pairs
ids = {}

# For each REFLACX id, store the corresponding DICOM id
for id, patient in reflacx_map.items():
    ids[id] = patient["dicom_id"]

# Load metadata to find images in MIMIC-CXR-JPG-2.0.0
with open("/mnt/thang_18tb/thang/physionet.org/files/gazetoy_metadata.json") as metadata:
    dat = json.load(metadata)

# Loop over each dicom_id and segment xray
with alive_bar(len(ids)) as bar:
    for reflacx_id, dicom_id in ids.items():
        # Obtain the model, img, prediction, and binary mask using the given file path
        model, img, pred, mask = segment(dat[dicom_id]["img_path_jpg"].replace("data_here", "/mnt/thang_18tb/thang/physionet.org/files"))

        # Logical operators require non-float dtype
        mask = mask.int()

        # Create full-chest segmentation mask with heart overlaying mediastanum, overlaying left lung, right lung, and diaphragm
        chest = mask[0, 8] * 3 # Heart
        chest = np.where(chest, chest, mask[0, 11] * 5) # Mediastanum
        chest = np.where(chest, chest, mask[0, 4]) # Left lung
        chest = np.where(chest, chest, mask[0, 5] * 2) # Right lung
        chest = np.where(chest, chest, mask[0, 10] * 4) # Diaphragm
        
        # Create lung-only segmentation mask with left lung overlaying right lung
        lungs = mask[0, 4] # Left lung
        lungs = np.where(lungs, lungs, mask[0, 5] * 2) # Right lung

        # Save full-chest and lung-only segmentation masks to pngs
        cv2.imwrite("masks/" + reflacx_id + "-" + dicom_id + ".png", chest)
        cv2.imwrite("masks/" + reflacx_id + "-" + dicom_id + "-lungs.png", lungs)

        bar()

