#!/usr/bin/env bash

#SBATCH --job-name=ocr_vqa
#SBATCH --output=logs_embedding_slurm/ocr_vqa_array_%A_%a.out  # Separate log for each task
#SBATCH --time=10-00:00:00
#SBATCH --partition=all
#SBATCH --nodes=1

# ==============================================================================
# == CONFIGURATION: Set the total number of subsets and parallel jobs         ==
# ==============================================================================

# 1. Define the total number of subsets you have (e.g., 0 to 3 for 4 subsets)

# 2. Define how many of those jobs can run IN PARALLEL at any one time.
#    This should match the number of GPUs you want to use.
#    For example, use %2 to run on 2 GPUs, %4 to run on 4 GPUs.
#SBATCH --array=1,3%2

# 3. Request resources for EACH INDIVIDUAL TASK in the array.
#    Each task needs 1 GPU.
#SBATCH --gpus-per-task=1

# Let Slurm pick any node from the list that can satisfy the --gpus-per-task request
#SBATCH --nodelist=worker-9,worker-2,worker-4,worker-5,worker-6,worker-7

echo "==================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
# Slurm automatically sets this to the correct single GPU for this task
echo "Assigned GPU: $CUDA_VISIBLE_DEVICES"
echo "==================================================================="

# Use an absolute path for your conda installation for robustness
# Replace `/path/to/your/miniconda3` with the actual path
conda activate autoseg

# --- Define Base Paths (using absolute paths is recommended) ---
BASE_SUBSET_DIR="./videos/ocr_vqa/subsets"
BASE_OUTPUT_DIR="./output/ocr_vqa"
BATCH_SIZE=8

# --- Determine the specific paths for THIS task using its array ID ---
# The $SLURM_ARRAY_TASK_ID variable is automatically provided by Slurm
TASK_ID_FORMATTED=$(printf "%03d" $SLURM_ARRAY_TASK_ID)

INPUT_SUBSET_DIR="$BASE_SUBSET_DIR/subset_$TASK_ID_FORMATTED"
OUTPUT_DIR="$BASE_OUTPUT_DIR/subset_$TASK_ID_FORMATTED"

# Create the task-specific output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Task is processing:"
echo "  Input Dir:  $INPUT_SUBSET_DIR"
echo "  Output Dir: $OUTPUT_DIR"

# --- Execute the Python script for this single task ---
# No need for backgrounding (&) or manual CUDA_VISIBLE_DEVICES.
# Slurm handles the parallelism and GPU assignment.
python auto-embedding-batch.py \
    --video_path "$INPUT_SUBSET_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE

# Check the exit code of the Python command
STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo "Task $SLURM_ARRAY_TASK_ID completed successfully."
else
    echo "Task $SLURM_ARRAY_TASK_ID failed with exit code $STATUS." >&2
fi

echo "==================================================================="

exit $STATUS