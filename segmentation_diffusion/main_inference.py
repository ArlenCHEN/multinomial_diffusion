import torch
import numpy as np
import pickle
import argparse
from diffusion_utils.utils import add_parent_path
import matplotlib.pyplot as plt
from matplotlib import gridspec
from eval_conf import eval_cfg
from datasets.data import get_data, get_data_id, add_data_args, get_plot_transform, get_plot_transform_1
from model import get_model, get_model_id, add_model_args
from diffusion_utils.base import DataParallelDistribution
from scipy.interpolate import griddata
from copy import deepcopy

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

class SePaint():
    def __init__(self, cfg, num_classes, data_shape):
        super(SePaint, self).__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.model = get_model(cfg, data_shape=data_shape)
        
        if cfg.parallel == 'dp':
            self.model = DataParallelDistribution(self.model)

        if torch.cuda.is_available():
                checkpoint = torch.load(path_check)
        else:
            checkpoint = torch.load(path_check, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.plot_trasform = get_plot_transform(cfg)

    def predict(self, gt_img, mask):
        model_kwargs = {}
        model_kwargs['input'] = gt_img
        model_kwargs['gt'] = gt_img
        model_kwargs['gt_mask'] = mask

        sample_fn = self.model.p_sample_loop_inpa

        result = sample_fn(
            model_kwargs=model_kwargs,
            eval_cfg=eval_cfg
        )
        result = result.cpu()
        result_np = result.squeeze().numpy()
        gt_np = gt_img.cpu().squeeze().numpy()

        sepaint_miou = compute_mIoU(result_np, gt_np, num_classes=num_classes)
        sepaint_acc = compute_acc(result_np, gt_np)

        print('sepaint miou: ', sepaint_miou)
        print('sepaint acc: ', sepaint_acc)

        # Generate masked image
        data_input = deepcopy(gt_img)
        data_input[mask==0] = 255
        input_color = self.plot_transform(data_input).to(torch.uint8)
        result[result==15] = 0

        colored_result = self.plot_transform(result).to(torch.uint8)
        gt_color = self.plot_transform(gt_img).to(torch.uint8)
        
        input_color_np = input_color[0].detach().cpu().numpy()
        colored_result_np = colored_result[0].detach().cpu().numpy()
        gt_color_np = gt_color[0].detach().cpu().numpy()

        input_color_np = np.transpose(input_color_np, (1,2,0))
        colored_result_np = np.transpose(colored_result_np, (1,2,0))
        gt_color_np = np.transpose(gt_color_np, (1,2,0))

        result_data = {
            'miou': sepaint_miou,
            'acc': sepaint_acc,
            'masked_img': input_color_np,
            'gt_img': gt_color_np,
            'prediction': colored_result_np
        }
        return result_data
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--nrow', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--double', type=eval, default=False)
    eval_args = parser.parse_args()

    path_args = '{}/args.pickle'.format(eval_args.model)
    path_check = '{}/check/checkpoint.pt'.format(eval_args.model)

    # Below two lines are only for nuscenes
    data_shape = (1, 100, 100)
    num_classes = 14

    with open(path_args, 'rb') as f:
        args = pickle.load(f)
    
    sepaint = SePaint(args, num_classes, data_shape)

    # ========= ! Modify here ! =========
    # TODO: Provide gt_img and mask as torch tensors and follow the torch format: [batch_size, channel, h, w]
    gt_img_tensor = None
    mask_tensor = None
    # ========= ! Modify here ! =========

    result_data = sepaint.predict(gt_img_tensor, mask_tensor)

    # TODO: Extract information you want :)
    