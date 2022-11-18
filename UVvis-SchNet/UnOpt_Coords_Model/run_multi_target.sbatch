#!/bin/bash
#SBATCH -p dlt_shared
#SBATCH -A mars_pli
#SBATCH -J UnOptSch
#SBATCH -t 4-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -o UnOpt_SchNet-%A.out
#SBATCH -e UnOpt_SchNet-%A.err

module purge
module load gcc/7.1.0
module load cuda/11.1
module load python/anaconda3.2019.3

source activate /people/mcna892/.conda/envs/g4mp2_ml

python train_models_multi.py -d cuda

