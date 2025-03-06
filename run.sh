#!/bin/bash -l
#
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:45:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
module load nvhpc
module load cuda/12.4.1
./build/run.out
