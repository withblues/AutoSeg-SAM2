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
from torch.utils.data import Dataset, DataLoader

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


class ImageDataset(Dataset):
    def __init__(self, image_names, image_dir):
        self.image_names = image_names
        self.image_dir = image_dir

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(image_path)

        if image is None:
            logger.warning(f"Failed to load image: {image_path}. Skipping.")
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image, image_name

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]

    if not batch:
        return None, None
    
    images, names = zip(*batch)
    
    return list(images), list(names)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path",type=str,required=True)
    parser.add_argument("--output_dir",type=str,required=True)
    parser.add_argument("--pred_iou_thresh",type=float,default=0.7)
    parser.add_argument("--box_nms_thresh",type=float,default=0.7)
    parser.add_argument("--stability_score_thresh",type=float,default=0.85)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--points_per_batch", type=int, default=64)
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


    dataset = ImageDataset(image_names=image_names, image_dir=image_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=2,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )

    lmdb_path = os.path.join(args.output_dir, f"masks.lmdb")
    env = lmdb.open(lmdb_path, map_size=1099511627776)
    
    for image_batch, batch_of_names in tqdm(dataloader):
        if image_batch is None:
            continue

        # generate mask
        masks, image_encoder_data = mask_generator.generate(image_batch)

        with env.begin(write=True) as txn:
            for idx, image_name in enumerate(batch_of_names):
                logger.info(f"Saving masks for image: {image_name}")
                save_dict = {}

                mask_records = []
                for mask in masks[idx]:
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
                    
                    save_dict['mask_l'] = mask_records

                save_dict["image_encoder"] = image_encoder_data[idx]

                value = dumps_npz(save_dict)
                txn.put(key=image_name.encode('utf-8'), value=value)
        
        del masks, image_encoder_data, image_batch
        torch.cuda.empty_cache()

    logger.info("Processing complete.")