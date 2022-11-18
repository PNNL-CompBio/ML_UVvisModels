#!/bin/bash
#SBATCH -p dlv
#SBATCH -A mulcme
#SBATCH -J g2mp4_test
#SBATCH -t 4-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -o slurm-%A.out
#SBATCH -e slurm-%A.err

module purge
module load gcc/7.1.0
module load cuda/11.1
#module load python/anaconda3.2019.3


source activate chemprop_IR

python train.py --data_path ./test_preds.csv  --dataset_type spectra --save_dir ./IR_test



