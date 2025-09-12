import os
from segment_anything_fast import sam_model_fast_registry, SamAutomaticMaskGenerator
import os
import torch
import numpy as np
import cv2
import argparse
from loguru import logger
import lmdb
import msgpack
import io
from tqdm import tqdm
from torch.utils.data import Dataset as TorchDataset, DataLoader
import msgpack
import msgpack_numpy as m
import zstandard as zstd
import json
m.patch()



def dumps_msgpack(data):
    return msgpack.packb(data, use_bin_type=True)

def dumps_msgpack_zstd(data):
    """Serializes with msgpack, then compresses with Zstandard."""
    packed_data = dumps_msgpack(data)
    return zstd.compress(packed_data)

def dumps_npz(data_dict, compress=True):
    with io.BytesIO() as buffer:
        if compress:
            np.savez_compressed(buffer, **data_dict, allow_pickle=True)
        else:
            np.save(buffer, **data_dict, allow_pickle=True)
        return buffer.getvalue()


class ImageDataset(TorchDataset):
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
    parser.add_argument("--batch_size", type=int, default=1)
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
    sam = sam_model_fast_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
    print(f'loaded model')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        # points_per_side=32,
        # pred_iou_thresh=args.pred_iou_thresh, 
        # box_nms_thresh=args.box_nms_thresh, 
        # stability_score_thresh=args.stability_score_thresh, 
        # min_mask_region_area=100,
        # mask_type='d',
        output_mode='coco_rle',
        process_batch_size=4,
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


    #### LMDB
    # embeddings lmdb
    # embed_lmdb_path = os.path.join(args.output_dir, "embeddings.lmdb")
    # env_embed = lmdb.open(embed_lmdb_path, map_size=1099511627776, writemap=True)

    # mask lmdb
    masks_lmdb_path = os.path.join(args.output_dir, "masks.lmdb")
    env_masks = lmdb.open(masks_lmdb_path, map_size=1099511627776, writemap=True)

    write_batch_size = 128
    embed_buffer = []
    mask_buffer = []

    # lmdb mapper
    mapper = {}
    subset = os.path.basename(args.output_dir)

    for image_batch, name_batch in tqdm(dataloader):
        if image_batch is None:
            continue

        image = image_batch[0]
        image_name =name_batch[0]

        # generate mask
        masks, _ = mask_generator.generate(image)

        mask_records = []
        for mask in masks:
            mask_data = {
                'segmentation': mask['segmentation'],
                'area': np.uint32(mask['area']),
                'bbox': np.array(mask['bbox'], dtype=np.int32),
                'predicted_iou': np.float16(mask['predicted_iou']),
                'point_coords': np.array(mask['point_coords'], dtype=np.float16),
                'stability_score': np.float16(mask['stability_score']),
                'crop_box': np.array(mask['crop_box'], dtype=np.int32),
                'embedding': np.array(mask['embedding'], dtype=np.float32) if 'embedding' in mask else None,
            }
            mask_records.append(mask_data)
            
            
        mask_save_dict = {'masks': mask_records}
        mask_value = dumps_msgpack_zstd(mask_save_dict)
        mask_buffer.append({'key': image_name, 'value': mask_value})

        #embed_value = image_encoder_data.astype(np.float16).tobytes()
        #embed_buffer.append({'key': image_name, 'value': embed_value})

        # save data 
        if len(mask_buffer) >= write_batch_size:
            logger.info(f"Saving masks")
            with env_masks.begin(write=True) as txn_masks:
                 #env_embed.begin(write=True) as txn_embed:
                
                for item in mask_buffer:
                    txn_masks.put(key=item['key'].encode('utf-8'), value=item['value'])
                
                # for item in embed_buffer:
                #     txn_embed.put(key=item['key'].encode('utf-8'), value=item['value'])

            # clear buffers
            mask_buffer = []
            embed_buffer = []

            logger.info(f"Done saving masks")

        # add image key to mapper
        mapper[image_name] = subset

        del masks, image_batch
        torch.cuda.empty_cache()

    # save last batch
    if mask_buffer:
        logger.info(f"Saving last buffer")
        with env_masks.begin(write=True) as txn_masks:
            
            for item in mask_buffer:
                txn_masks.put(key=item['key'].encode('utf-8'), value=item['value'])
            
            # for item in embed_buffer:
            #     txn_embed.put(key=item['key'].encode('utf-8'), value=item['value'])

    logger.info("saving mapper json")
    with open(os.path.join(args.output_dir, 'map.json'), 'w') as fp:
        json.dump(mapper, fp, indent=2)

    env_masks.close()
    #env_embed.close()
    logger.info("Processing complete. All databases closed.")