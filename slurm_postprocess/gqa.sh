#!/usr/bin/env bash

#SBATCH --job-name=gqa
#SBATCH --output=logs_postprocess_slurm/gqa_array_%A_%a.out  # Separate log for each task
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
#SBATCH --array=0-3%4


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
BASE_SUBSET_DIR="./videos/gqa/subsets"
BASE_MASK_DIR="./output/gqa"
BASE_OUTPUT_DIR="./postprocess/gqa"

# --- Determine the specific paths for THIS task using its array ID ---
# The $SLURM_ARRAY_TASK_ID variable is automatically provided by Slurm
TASK_ID_FORMATTED=$(printf "%03d" $SLURM_ARRAY_TASK_ID)

INPUT_SUBSET_DIR="$BASE_SUBSET_DIR/subset_$TASK_ID_FORMATTED"
INPUT_MASK_DIR="$BASE_MASK_DIR/subset_$TASK_ID_FORMATTED/masks.lmdb"
OUTPUT_DIR="$BASE_OUTPUT_DIR/subset_$TASK_ID_FORMATTED"

# Create the task-specific output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Task is processing:"
echo "  Input Dir:  $INPUT_SUBSET_DIR"
echo "  Output Dir: $OUTPUT_DIR"

# --- Execute the Python script for this single task ---
# No need for backgrounding (&) or manual CUDA_VISIBLE_DEVICES.
# Slurm handles the parallelism and GPU assignment.
python postprocess.py \
    --image_path "$INPUT_SUBSET_DIR" \
    --mask_path "$INPUT_MASK_DIR" \
    --output_dir $OUTPUT_DIR

# Check the exit code of the Python command
STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo "Task $SLURM_ARRAY_TASK_ID completed successfully."
else
    echo "Task $SLURM_ARRAY_TASK_ID failed with exit code $STATUS." >&2
fi

echo "==================================================================="

exit $STATUS