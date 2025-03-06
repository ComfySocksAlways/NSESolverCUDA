#!/bin/bash -l
#
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:05:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
module load nvhpc
module load cuda
module load gcc/12.1.0
nsys profile --stats=true build/run.out