import os
from pathlib import Path
import json
import zlib
import cv2
from PIL import Image
import torch

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import lmdb # pip install lmdb

import segmentation_diffusion.nuscenes.nuscenes_utils as nu
from segmentation_diffusion.nuscenes.nuscenes_utils import bytes_to_array

mask_path = '/home/zheng/Softwares/RePaint/data/datasets/gt_keep_masks/thick/000015.png'
is_extra_mask = True # Same setting as the 2D Cityscapes if set as True

dataroot = Path('/media/zheng/Cookie/Datasets/Segmentation/nuScenes').resolve()

# # small samples are for sparse raw bev map: only for testing
# # Note that the metadata is different from the complete data
# gt_db_path = dataroot / Path('lmdb/small_samples/GT_BEV_CAM_FRONT')
# raw_db_path = dataroot / Path('lmdb/small_samples/SPARSE_RAW_BEV_CAM_FRONT')
# nusc_metadata_path = dataroot / Path('v1.0-mini-CAM_FRONT_token.json')

# Use samples when training
# GT_BEV_CAM_FRONT_2 is the corrected version of the data
gt_db_path = dataroot / Path('lmdb/samples/GT_BEV_CAM_FRONT_2')
raw_db_path = dataroot / Path('lmdb/samples/RAW_BEV_CAM_FRONT')
nusc_metadata_path = dataroot / Path('v1.0-trainval-meta-custom.json')

# Only true when small_samples are used
is_small = False

# RGB info
nusc_idx_to_color = {
    0: (0, 207, 191),
    1: (175, 0, 75),
    2: (75, 0, 75),
    3: (112, 180, 60),
    4: (255, 158, 0),
    5: (255, 99, 71),
    6: (255, 69, 0),
    7: (255, 140, 0),
    8: (233, 150, 70),
    9: (138, 43, 226),
    10: (255, 61, 99),
    11: (220, 20, 60),
    12: (47, 79, 79),
    13: (112, 128, 144)
}
color_map = {i + 1: c for i, c in nusc_idx_to_color.items()}

# Please make changes in the function of make_composite in nuscenes_utils.py if you want to remap the id
color_map[0] = [255, 255, 255]
color_map[4] = [255, 255, 255]  # do not color "terrain"
# color_map[15] = [255, 255, 255]  # uncomment this line to not color "lidar mask"
color_map[255] = [50, 50, 50]

def nuscenes_indices_segmentation_to_img(img):
    rgb = nu.color_components(img, color_map=color_map)
    rgb_tensor = torch.from_numpy(rgb) # Convert numpy array to torch tensor
    rgb_tensor = rgb_tensor.permute(0, 3, 1, 2)
    return rgb_tensor

class nuScenes():
    def __init__(self, split='train', resolution=(100, 100), transform=None):
        H, W = resolution
        
        self.dataroot = dataroot
        self.gt_db_path = gt_db_path
        self.raw_db_path = raw_db_path
        self.nusc_metadata_path = nusc_metadata_path
        self.split = split
        self.resolution = resolution
        self.transform = transform
        
        with open(nusc_metadata_path, 'r') as f:
            self.nusc_metadata = json.load(f)
        
        self.splits_dir = self.dataroot / 'splits'
        
        if split == 'train':
            self.split_path = self.splits_dir / 'train_roddick.txt'
            with open(self.split_path, 'r') as f:
                self.data_samples = f.read().split() # Training data samples
            
            print('Number of training samples: ', len(self.data_samples))
        elif split == 'test':
            self.split_path = self.splits_dir / 'val_roddick.txt'
            with open(self.split_path, 'r') as f:
                self.data_samples = f.read().split()[200:] # Validation data samples
            
            print('Number of validation samples: ', len(self.data_samples))

        if not is_small:
            self.sample_token_to_cam_token = {}

            for scene in self.nusc_metadata.values():
                for sample_token, sample in scene['scene_data'].items():
                    self.sample_token_to_cam_token[sample_token] = sample['CAM_FRONT']['token']
        
        # Read the LMDB data files
        self.gt_db = lmdb.open(path=str(gt_db_path), readonly=True, lock=False)
        
        # Read raw map only when testing
        if self.split == 'test':
            self.raw_db = lmdb.open(path=str(raw_db_path), readonly=True, lock=False)

    def db_value_to_array(self, value):
        value_unzipped = zlib.decompress(value)
        return bytes_to_array(value_unzipped)

    def __getitem__(self, index):
        if is_small:
            cam_token = self.nusc_metadata[index] # small data; already the cam_token
        else:
            sample_token = self.data_samples[index] # complete data reading
            cam_token = self.sample_token_to_cam_token[sample_token]
        
        cam_token_bytes = str.encode(cam_token, 'utf-8')
        
        with self.gt_db.begin() as txn:
            value = txn.get(cam_token_bytes)
            # shape: (15, 196, 200)
            bev_gt = self.db_value_to_array(value)
        
        if self.split == 'test':
            with self.raw_db.begin() as txn:
                value = txn.get(cam_token_bytes)
                # shape: (14, 200, 200)
                bev_raw = self.db_value_to_array(value)
        
        # Get id images from logit, where the last channel (lidar mask) is removed
        # Otherwise, the model will learn the distribution of lidar mask as id of 15 and ignore the real ids under lidar mask
        bev_gt_id_np = nu.make_composite(bev_gt[:-1,:,:]).astype(int)
        
        if self.split == 'test':
            bev_raw_id_np = nu.make_composite(bev_raw).astype(int)
        
        # Downsample images to a specified resolution
        bev_gt_id_np = cv2.resize(bev_gt_id_np, self.resolution, interpolation=cv2.INTER_NEAREST)
        
        if self.split == 'test':
            bev_raw_id_np = cv2.resize(bev_raw_id_np, self.resolution, interpolation=cv2.INTER_NEAREST)
            # Add .copy if np.flip is used
            bev_raw_id_np = np.flip(bev_raw_id_np, axis=0).copy()
            bev_mask_np = np.ones(bev_raw_id_np.shape).astype(int)
            # 
            mask_pos = np.where((bev_raw_id_np==0) | (bev_raw_id_np==4))
            bev_mask_np[mask_pos[0], mask_pos[1]] = 0 
            
            if is_extra_mask:
                temp_mask = Image.open(mask_path)
                temp_mask.load()
                temp_mask = temp_mask.resize((self.resolution[1], self.resolution[0]), Image.NEAREST)          
                # Extract a slice of the mask. np.array(temp_mask) has a shape of [resolution[1], resolution[0], 3]
                temp_mask_arr = np.array(temp_mask)[:,:,0].astype(np.float32)/255.0
                bev_mask_np = temp_mask_arr
        
        # Only apply the transform when training
        if self.split == 'train':
            if self.transform:
                bev_gt_id_img = Image.fromarray(bev_gt_id_np.astype('uint8'))
                trans_bev_gt_id_img = self.transform(bev_gt_id_img)
                trans_bev_gt_id_np = np.array(trans_bev_gt_id_img)
                trans_bev_gt_id_tensor = torch.tensor(trans_bev_gt_id_np).long()
                bev_gt_id_tensor = trans_bev_gt_id_tensor.unsqueeze(0)
            else:
                bev_gt_id_tensor = torch.tensor(bev_raw_id_np).long().unsqueeze(0)
        else:
            bev_gt_id_tensor = torch.tensor(bev_gt_id_np).long().unsqueeze(0)
            bev_raw_id_tensor = torch.tensor(bev_raw_id_np).long().unsqueeze(0)
            bev_mask_tensor = torch.tensor(bev_mask_np).long().unsqueeze(0)
        
        is_plot = False
        if is_plot:
            # Get the color image
            bev_gt_color_np = nu.color_components(bev_gt_id_np, color_map=color_map)
            if self.split == 'test':
                bev_raw_color_np = nu.color_components(bev_raw_id_np, color_map=color_map)
            
            fig, axs = plt.subplots(ncols=2, figsize=(20, 7))
            ax = axs[0]
            ax.imshow(np.flip(bev_gt_color_np, 0))
            ax.set_title('map based BEV')
            ax.axis('off')

            # Only show the raw bev when testing
            if self.split == 'test':
                ax = axs[1]
                ax.imshow(bev_raw_color_np)
                ax.invert_yaxis()
                ax.set_title('lidarseg BEV')

            legend_colors = [np.append(np.array(nusc_idx_to_color[idx]) / 255, 1) for idx in range(len(nusc_idx_to_color))]
            patches = [mpatches.Patch(color=legend_colors[i], label=label)
                    for i, label in enumerate(nu.NUSC_LIDAR_CLASS_NAMES) if i not in [1, 3]]
            ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            ax.axis('off')
            
            plt.show()
        
        # Return different data for testing than training
        if self.split == 'test':
            if is_extra_mask:
                return {
                    'input': bev_gt_id_tensor,
                    'gt': bev_gt_id_tensor,
                    'gt_mask': bev_mask_tensor
                }
            else:
                return {
                    'input': bev_raw_id_tensor,
                    'gt': bev_gt_id_tensor,
                    'gt_mask': bev_mask_tensor
                }
        else:
            return bev_gt_id_tensor

    def __len__(self):
        return len(self.data_samples)
            