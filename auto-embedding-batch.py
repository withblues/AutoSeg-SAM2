import os
import torch
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import argparse
from loguru import logger
import lmdb
import msgpack
import io

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True




def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.
    
    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou

            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    # If there are no masks with scores above threshold, the top 3 masks are selected
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    # import ipdb; ipdb.set_trace()
    return selected_idx

def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep

def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    masks_new = ()
    for masks_lvl in (args):
        seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new
        
def dumps_msgpack(data):
    return msgpack.dumps(data, use_bin_type=True)

def dumps_npz(data_dict, compress=True):
    with io.BytesIO() as buffer:
        if compress:
            np.savez_compressed(buffer, **data_dict, allow_pickle=True)
        else:
            np.save(buffer, **data_dict, allow_pickle=True)
        return buffer.getvalue()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path",type=str,required=True)
    parser.add_argument("--output_dir",type=str,required=True)
    parser.add_argument("--level",choices=['default','small','middle','large'])
    parser.add_argument("--batch_size",type=int,default=20)
    parser.add_argument("--postnms",type=int,default=1)
    parser.add_argument("--pred_iou_thresh",type=float,default=0.7)
    parser.add_argument("--box_nms_thresh",type=float,default=0.7)
    parser.add_argument("--stability_score_thresh",type=float,default=0.85)
    args = parser.parse_args()
    logger.add(os.path.join(args.output_dir,f'{args.level}.log'), rotation="500 MB")
    logger.info(args)
    image_dir = args.video_path
    level = args.level
    base_dir = args.output_dir

    ##### load Sam1 Model #####
    sam_ckpt_path="checkpoints/sam1/sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=args.pred_iou_thresh, 
        box_nms_thresh=args.box_nms_thresh, 
        stability_score_thresh=args.stability_score_thresh, 
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )
    # scan all the JPEG frame names in this directory
    image_names = [
        p for p in os.listdir(image_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]

    lmdb_path = os.path.join(args.output_dir, f"{args.level}_masks.lmdb")
    env = lmdb.open(lmdb_path, map_size=1099511627776)

    with env.begin(write=True) as txn:
        for idx, image_name in enumerate(image_names):
            logger.info(f"Processing image {idx + 1}/{len(image_names)}: {image_name}")

            image_path = os.path.join(image_dir, image_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # generate mask
            masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image)

            if args.postnms:
                masks_default, masks_s, masks_m, masks_l = \
                    masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
            if level == 'default':
                masks = [mask for mask in masks_default]
                other_masks = [mask for mask in masks_l] + [mask for mask in masks_m] + [mask for mask in masks_s] 
            elif level == 'small':
                masks = [mask for mask in masks_s]
                other_masks = None
            elif level == 'middle':
                masks = [mask for mask in masks_m]
                other_masks = [mask for mask in masks_s]
            elif level == 'large':
                masks = [mask for mask in masks_l]
                other_masks = [mask for mask in masks_s] + [mask for mask in masks_m]
            else:
                raise NotImplementedError
        
            mask_records = []
            for mask in masks:
                mask_data = {
                    'segmentation': mask['segmentation'].astype(np.bool_),
                    'area': mask['area'],
                    'bbox': mask['bbox'],
                    'predicted_iou': mask['predicted_iou'],
                    'point_coords': mask['point_coords'],
                    'stability_score': mask['stability_score'],
                    'crop_box': mask['crop_box'],
                    'embeddings': mask['embeddings'].astype(np.float32) if 'embeddings' in mask else None,
                }
                mask_records.append(mask_data)

            value = dumps_npz({'masks': mask_records})
            txn.put(key=image_name.encode('utf-8'), value=value)