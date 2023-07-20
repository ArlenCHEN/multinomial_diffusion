# import os
# import math
import torch
import numpy as np
import pickle
import argparse
# import torchvision.utils as vutils
from diffusion_utils.utils import add_parent_path
# import torchvision
# import imageio
import matplotlib.pyplot as plt
from matplotlib import gridspec

from eval_conf import eval_cfg

# Data
add_parent_path(level=1)
from datasets.data import get_data, get_data_id, add_data_args, get_plot_transform, get_plot_transform_1

# Model
from model import get_model, get_model_id, add_model_args
from diffusion_utils.base import DataParallelDistribution

from scipy.interpolate import griddata


def compute_acc(pred, target):
    corrects = (pred == target).astype(float)
    acc = np.mean(corrects)
    acc = acc * 100
    return acc
    
def compute_mIoU(pred, target, num_classes):
    """
    Computes the mean Intersection over Union (mIoU) for two multiclass segmentation images.
    
    Args:
    - pred: predicted segmentation image, numpy array of shape (height, width)
    - target: ground truth segmentation image, numpy array of shape (height, width)
    - num_classes: number of classes in the segmentation images
    
    Returns:
    - mIoU: mean Intersection over Union (mIoU) value
    """
    mIoU = 0.0
    for class_id in range(num_classes):
        pred_mask = pred == class_id
        target_mask = target == class_id
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        
        if union == 0:
            class_miou = 1.0 / num_classes
        else:
            class_miou = float(intersection) / float(union)
        mIoU += class_miou
        result = 100*(mIoU / num_classes)
    return result

def grid2tensor(grid):
    # Get the position of nan values
    nan_pos = np.argwhere(np.isnan(grid))
    
    if len(nan_pos): # if there are nans
        # print('nan pos: ', nan_pos)
        grid[nan_pos[:, 0], nan_pos[:, 1]] = 0 # assign 1 (Road) to nan-valued positions; 0 is the 255 label
    else: # if there is no nan
        print('No nan there...')
        
    # Make sure the label is int
    grid = grid.astype(int)
    
    grid_tensor = torch.from_numpy(grid)
    grid_tensor = grid_tensor.unsqueeze(0)
    grid_tensor = grid_tensor.unsqueeze(0)
    
    return grid_tensor

###########
## Setup ##
###########

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--samples', type=int, default=64)
parser.add_argument('--nrow', type=int, default=8)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--double', type=eval, default=False)
eval_args = parser.parse_args()

path_args = '{}/args.pickle'.format(eval_args.model)
path_check = '{}/check/checkpoint.pt'.format(eval_args.model)

# NOTE: Please uncomment this line if you want to have same result for different runs
# torch.manual_seed(eval_args.seed)

###############
## Load args ##
###############

with open(path_args, 'rb') as f:
    args = pickle.load(f)

##################
## Specify data ##
##################

# train_loader is not used in this file; args is loaded from the save trained args
train_loader, eval_loader, data_shape, num_classes = get_data(args)

###################
## Specify model ##
###################

model = get_model(args, data_shape=data_shape)
if args.parallel == 'dp':
    model = DataParallelDistribution(model)

if torch.cuda.is_available():
    checkpoint = torch.load(path_check)
else:
    checkpoint = torch.load(path_check, map_location='cpu')
model.load_state_dict(checkpoint['model'])

if torch.cuda.is_available():
    model = model.cuda()

print('Loaded weights for model at {}/{} epochs'.format(checkpoint['current_epoch'], args.epochs))
print('In inpaint_sample, args.dataset: ', args.dataset)

plot_transform = get_plot_transform(args)

if args.dataset == 'cityscapes_fine_large':
    plot_transform_1 = get_plot_transform_1(args)
    
# Load testing data
for minibatch_data in eval_loader:
    model_kwargs = {}
    
    data_input = minibatch_data['input']
    gt = minibatch_data['gt']
    gt_mask = minibatch_data['gt_mask']

    # # ==================== ACDC data ====================   
    # data_root = '/home/zheng/Softwares/Experiments/diff/testing_data/cityscapes'
    # npy_path = os.path.join(data_root, 'GP020606_frame_000470_rgb_anon.npy')
    # mask_path = os.path.join(data_root, 'mask.npy')
    
    # resolution = (256, 128)
    # seg_logit = np.load(npy_path)
    # seg_id = np.argmax(seg_logit, axis=0)
    # print(seg_id)
    # seg_id = cv2.resize(seg_id, resolution, interpolation=cv2.INTER_NEAREST)
    # mask = np.load(mask_path)
    # mask = cv2.resize(mask, resolution, interpolation=cv2.INTER_NEAREST)
    
    # input = torch.tensor(seg_id).unsqueeze(0).unsqueeze(0).long()
    # gt = input
    # gt_mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0).long()
    
    model_kwargs['input'] = data_input
    model_kwargs['gt'] = gt
    model_kwargs['gt_mask'] = gt_mask
     
    input_id_np = data_input[0][0].detach().cpu().numpy()
    
    # shape: (1, h, w)
    gt_mask_np = gt_mask[0][0].detach().cpu().numpy()
    
    img_h = input_id_np.shape[0]
    img_w = input_id_np.shape[1]
    grid_x, grid_y = np.mgrid[0:1:complex(False, img_h), 0:1:complex(False, img_w)]
    
    valid_pos = np.where(gt_mask_np==1.)
    values = input_id_np[gt_mask_np==1.]    
    
    # Covert the coordinates of points to be within [0, 1]    
    points = np.hstack((valid_pos[0][:, np.newaxis].astype(float)/img_h, valid_pos[1][:, np.newaxis].astype(float)/img_w))

    # nan values can be generated by methods OTHER THAN 'nearest'
    nearest_grid = griddata(points, values, (grid_x, grid_y), method='nearest')
    linear_grid = griddata(points, values, (grid_x, grid_y), method='linear')
    cubic_grid = griddata(points, values, (grid_x, grid_y), method='cubic')
    
    if args.dataset == 'cityscapes_fine_large':
        # Limit the value of cubic_grid less than 35
        cubic_grid[cubic_grid > 33] = 33
        cubic_grid[cubic_grid < -1] = -1
    elif args.dataset == 'nuscenes':
        cubic_grid[cubic_grid > 14] = 14
        cubic_grid[cubic_grid < 0] = 0
    
    nearest_grid_tensor = grid2tensor(nearest_grid)
    linear_grid_tensor = grid2tensor(linear_grid)
    cubic_grid_tensor = grid2tensor(cubic_grid)
    
    sample_fn = model.p_sample_loop_inpa
    
    result = sample_fn(
        model_kwargs=model_kwargs,
        eval_cfg=eval_cfg
    )
    
    # The input to plot_transform must be torch tensor on cpu    
    result = result.cpu()
    gt = gt.cpu()
    data_input = data_input.cpu()
    
    result_np = result.squeeze().numpy()
    gt_np = gt.squeeze().numpy()
    
    # ======================= Compute metrics ========================
    if args.dataset == 'cityscapes_fine_large':
        num_classes = 35
    elif args.dataset == 'nuscenes':
        num_classes = 14
        
    sepaint_miou = compute_mIoU(result_np, gt_np, num_classes=num_classes)
    nearest_miou = compute_mIoU(nearest_grid, gt_np, num_classes=num_classes)
    linear_miou = compute_mIoU(linear_grid, gt_np, num_classes=num_classes)
    cubic_miou = compute_mIoU(cubic_grid, gt_np, num_classes=num_classes)
    
    sepaint_acc = compute_acc(result_np, gt_np)
    nearest_acc = compute_acc(nearest_grid, gt_np)
    linear_acc = compute_acc(linear_grid, gt_np)
    cubic_acc = compute_acc(cubic_grid, gt_np)
    
    print('sepaint miou: ', sepaint_miou)
    print('sepaint acc: ', sepaint_acc)
    
    print('nearest miou: ',nearest_miou)
    print('nearest acc: ', nearest_acc)

    print('linear miou: ', linear_miou)
    print('linear acc: ', linear_acc)
    
    print('cubic miou: ', cubic_miou)
    print('cubic acc: ', cubic_acc)
    
    # ======================= Colorize results ========================
    # Colorize the missing regions
    if args.dataset == 'cityscapes_fine_large':
        input_color = plot_transform_1(data_input, gt_mask).to(torch.uint8)
    elif args.dataset == 'nuscenes':
        data_input[gt_mask == 0] = 255
        input_color = plot_transform(data_input).to(torch.uint8)
        result[result==15] = 0
        
    nearest_grid_tensor_color = plot_transform(nearest_grid_tensor).to(torch.uint8)
    linear_grid_tensor_color = plot_transform(linear_grid_tensor).to(torch.uint8)
    cubic_grid_tensor_color = plot_transform(cubic_grid_tensor).to(torch.uint8)
    
    colored_result = plot_transform(result).to(torch.uint8)
    gt_color = plot_transform(gt).to(torch.uint8)
    
    # shape: (3, h, w)
    input_color_np = input_color[0].detach().cpu().numpy()
    nearest_grid_color_np = nearest_grid_tensor_color[0].detach().cpu().numpy()
    linear_grid_color_np = linear_grid_tensor_color[0].detach().cpu().numpy()
    cubic_grid_color_np = cubic_grid_tensor_color[0].detach().cpu().numpy()
    
    colored_result_np = colored_result[0].detach().cpu().numpy()
    gt_color_np = gt_color[0].detach().cpu().numpy()
    
    input_color_np = np.transpose(input_color_np, (1,2,0))
    nearest_grid_color_np = np.transpose(nearest_grid_color_np, (1,2,0))
    linear_grid_color_np = np.transpose(linear_grid_color_np, (1,2,0))
    cubic_grid_color_np = np.transpose(cubic_grid_color_np, (1,2,0))
    
    colored_result_np = np.transpose(colored_result_np, (1,2,0))
    gt_color_np = np.transpose(gt_color_np, (1,2,0))
    
    nrow = 2
    ncol = 3
    
    if args.dataset == 'cityscapes_fine_large':
        fig = plt.figure(figsize=(14, 7))
        
        gs = gridspec.GridSpec(nrow, ncol,
                        wspace=0.0, hspace=-0.56, 
                        top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
                        left=0.5/(ncol+1), right=1-0.5/(ncol+1))
        
    elif args.dataset == 'nuscenes':
        fig = plt.figure(figsize=(10, 10))
        
        gs = gridspec.GridSpec(nrow, ncol,
                        wspace=0.1, hspace=0.1, 
                        top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
                        left=0.5/(ncol+1), right=1-0.5/(ncol+1))
        
    ax1 = fig.add_subplot(gs[0,0])
    im1 = ax1.imshow(input_color_np)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    if args.dataset == 'nuscenes':
        ax1.invert_yaxis()
    ax1.axis('off')
    
    ax3 = fig.add_subplot(gs[0,1])
    im3 = ax3.imshow(gt_color_np)
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])
    if args.dataset == 'nuscenes':
        ax3.invert_yaxis()
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[0,2])
    im4 = ax4.imshow(colored_result_np)
    ax4.set_yticklabels([])
    ax4.set_xticklabels([])
    if args.dataset == 'nuscenes':
        ax4.invert_yaxis()
    ax4.axis('off')
    
    ax5 = fig.add_subplot(gs[1,0])
    im5 = ax5.imshow(nearest_grid_color_np)
    ax5.set_yticklabels([])
    ax5.set_xticklabels([])
    if args.dataset == 'nuscenes':
        ax5.invert_yaxis()
    ax5.axis('off')
    
    ax5 = fig.add_subplot(gs[1,1])
    im5 = ax5.imshow(linear_grid_color_np)
    ax5.set_yticklabels([])
    ax5.set_xticklabels([])
    if args.dataset == 'nuscenes':
        ax5.invert_yaxis()
    ax5.axis('off')
    
    ax5 = fig.add_subplot(gs[1,2])
    im5 = ax5.imshow(cubic_grid_color_np)
    ax5.set_yticklabels([])
    ax5.set_xticklabels([])
    if args.dataset == 'nuscenes':
        ax5.invert_yaxis()
    ax5.axis('off')
    
    plt.show()   