import os
import torch
import numpy as np
from segment_anything import sam_model_registry
from segment_anything.automatic_batched_mask_generator import SamAutomaticMaskGenerator
import cv2
import argparse
from loguru import logger
import lmdb
import msgpack
import io
from tqdm import tqdm

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def dumps_msgpack(data):
    return msgpack.dumps(data, use_bin_type=True)

def dumps_npz(data_dict, compress=True):
    with io.BytesIO() as buffer:
        if compress:
            np.savez_compressed(buffer, **data_dict, allow_pickle=True)
        else:
            np.save(buffer, **data_dict, allow_pickle=True)
        return buffer.getvalue()

def create_batches(data_list, batch_size):
    for i in range(0, len(data_list), batch_size):
        yield data_list[i:i + batch_size]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path",type=str,required=True)
    parser.add_argument("--output_dir",type=str,required=True)
    parser.add_argument("--pred_iou_thresh",type=float,default=0.7)
    parser.add_argument("--box_nms_thresh",type=float,default=0.7)
    parser.add_argument("--stability_score_thresh",type=float,default=0.85)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    # logging
    logger.add(os.path.join(args.output_dir, 'mask_generation.log'), rotation="500 MB")
    logger.info(args)

    # directories
    image_dir = args.video_path
    base_dir = args.output_dir

    ##### load Sam1 Model #####
    sam_ckpt_path="checkpoints/sam1/sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        points_per_batch=128,
        pred_iou_thresh=args.pred_iou_thresh, 
        box_nms_thresh=args.box_nms_thresh, 
        stability_score_thresh=args.stability_score_thresh, 
        min_mask_region_area=100,
    )

    # scan all the JPEG frame names in this directory
    image_names = [
        p for p in os.listdir(image_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]

    lmdb_path = os.path.join(args.output_dir, f"masks.lmdb")
    env = lmdb.open(lmdb_path, map_size=1099511627776)

    image_name_batches = create_batches(image_names, args.batch_size)

    with env.begin(write=True) as txn:
        for batch_of_names in tqdm(image_name_batches, total=len(image_names) // args.batch_size + 1):
            
            image_batch = []
            for image_name in batch_of_names:
                image_path = os.path.join(image_dir, image_name)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_batch.append(image)

            # generate mask
            masks = mask_generator.generate(image_batch)

            for image_name, masks_for_one_image in zip(batch_of_names, masks):
                logger.info(f"Saving masks for image: {image_name}")

                mask_records = []
                for mask in masks_for_one_image:
                    mask_data = {
                        'segmentation': mask['segmentation'].astype(np.bool_),
                        'area': mask['area'],
                        'bbox': mask['bbox'],
                        'predicted_iou': mask['predicted_iou'],
                        'point_coords': mask['point_coords'],
                        'stability_score': mask['stability_score'],
                        'crop_box': mask['crop_box'],
                        'embeddings': np.array(mask['embeddings'], dtype=np.float32) if 'embeddings' in mask else None,
                    }
                    mask_records.append(mask_data)

                value = dumps_npz({'masks': mask_records})
                txn.put(key=image_name.encode('utf-8'), value=value)

    logger.info("Processing complete.")