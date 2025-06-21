#!/usr/bin/env bash

#SBATCH --job-name=test
#SBATCH --output=test.out
#SBATCH --ntasks=1
#SBATCH --time=5-00:10:00
#SBATCH --partition=major
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=worker-2,worker-5,worker-6,worker-7,worker-9


conda activate autoseg


python auto-embedding-batch.py --video_path videos/chickenchicken --output_dir output/chickenchicken --batch_size 1

