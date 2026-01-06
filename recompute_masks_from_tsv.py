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
import msgpack_numpy as m
import zstandard as zstd
import pandas as pd
import base64

# Import SAM and its registry
from segment_anything_fast import sam_model_fast_registry, SamAutomaticMaskGenerator

# --- Serialization functions from your script ---
m.patch()

def dumps_msgpack(data):
    return msgpack.packb(data, use_bin_type=True)

def dumps_msgpack_zstd(data):
    """Serializes with msgpack, then compresses with Zstandard."""
    packed_data = dumps_msgpack(data)
    return zstd.compress(packed_data)

# --- New Dataset Class to read images from VLMEvalKit's TSV format ---
class TSVImageDataset(TorchDataset):
    """A PyTorch Dataset to load images directly from a Base64-encoded TSV file."""
    def __init__(self, tsv_path):
        logger.info(f"Loading TSV file from {tsv_path}...")
        self.df = pd.read_csv(tsv_path, sep='\t')
        logger.info(f"Found {len(self.df)} total entries in the TSV.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # The 'index' column from the TSV file will be our unique key
        key = str(row['index'])
        b64_string = row['image']

        try:
            image_bytes = base64.b64decode(b64_string)
            image_np_array = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("cv2.imdecode returned None, check image format.")
            # Convert from BGR (cv2 default) to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.warning(f"Failed to decode image for index {key}. Error: {e}. Skipping.")
            return None

        return image, key

def custom_collate_fn(batch):
    """Collate function to filter out None values from failed image decodings."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None
    images, keys = zip(*batch)
    return list(images), list(keys)

# --- Main script logic adapted from your auto-embedding-fast.py ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate and save SAM masks from a VLMEvalKit TSV file to an LMDB database.")
    parser.add_argument("--tsv_path", type=str, required=True, help="Path to the VLMEvalKit TSV file (e.g., COCO_VAL.tsv).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output 'masks.lmdb' file and logs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for loading data. SAM processes one image at a time from the batch.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.add(os.path.join(args.output_dir, 'mask_generation_from_tsv.log'), rotation="500 MB")
    logger.info(f"Starting mask generation with args: {args}")

    # 1. Load SAM Model (logic from your script)
    sam_ckpt_path = "checkpoints/sam1/sam_vit_h_4b8939.pth"
    sam = sam_model_fast_registry["vit_h"](checkpoint=sam_ckpt_path).to("cuda")
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        output_mode='coco_rle',
        process_batch_size=4,  # Internal batching for SAM
    )
    logger.info("SAM model and mask generator loaded successfully.")

    # 2. Open LMDB database
    masks_lmdb_path = os.path.join(args.output_dir, "masks.lmdb")
    env_masks = lmdb.open(masks_lmdb_path, map_size=1099511627776)

    # 3. Check for already processed images to allow resuming
    processed_keys = set()
    with env_masks.begin(write=False) as txn:
        cursor = txn.cursor()
        for key_bytes in cursor.iternext(values=False):
            processed_keys.add(key_bytes.decode('utf-8'))
    logger.info(f"Found {len(processed_keys)} already processed entries in the LMDB.")

    # 4. Create Dataset and DataLoader, filtering for unprocessed items
    full_dataset = TSVImageDataset(tsv_path=args.tsv_path)
    
    unprocessed_indices = [i for i, row in full_dataset.df.iterrows() if str(row['index']) not in processed_keys]
    if not unprocessed_indices:
        logger.info("All images from the TSV have already been processed. Exiting.")
        env_masks.close()
        exit()
        
    logger.info(f"Total images in TSV: {len(full_dataset)}. Processing {len(unprocessed_indices)} new images.")
    
    # Create a new DataFrame containing only the rows that need processing
    unprocessed_df = full_dataset.df.iloc[unprocessed_indices].copy()
    
    # We define a small inner class to wrap the filtered dataframe
    class FilteredTSVImageDataset(TSVImageDataset):
        def __init__(self, dataframe):
            self.df = dataframe
    
    dataset_to_process = FilteredTSVImageDataset(unprocessed_df)

    dataloader = DataLoader(
        dataset_to_process,
        batch_size=args.batch_size,
        num_workers=2,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )

    # 5. Process images and save to LMDB
    write_batch_size = 128
    mask_buffer = []

    for image_batch, key_batch in tqdm(dataloader, desc="Generating Masks"):
        if image_batch is None:
            continue

        for image, key in zip(image_batch, key_batch):
            try:
                masks, _ = mask_generator.generate(image)
            except RuntimeError as e:
                logger.error(f"Skipping image with key {key} due to a runtime error: {e}")
                continue

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
            mask_buffer.append({'key': key, 'value': mask_value})

            # Write to LMDB in batches
            if len(mask_buffer) >= write_batch_size:
                with env_masks.begin(write=True) as txn_masks:
                    for item in mask_buffer:
                        txn_masks.put(key=item['key'].encode('utf-8'), value=item['value'])
                logger.info(f"Saved a batch of {len(mask_buffer)} masks to LMDB.")
                mask_buffer = []

    # Save any remaining items in the buffer
    if mask_buffer:
        with env_masks.begin(write=True) as txn_masks:
            for item in mask_buffer:
                txn_masks.put(key=item['key'].encode('utf-8'), value=item['value'])
        logger.info(f"Saved the final batch of {len(mask_buffer)} masks to LMDB.")

    env_masks.close()
    logger.info("Processing complete. LMDB database is updated.")
