#!/bin/bash
#SBATCH --job-name=cifar10_e16

#SBATCH 
#SBATCH --partition=PA100q
#SBATCH -w node12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:8
#SBATCH --output=%j_cifar10_e16_T.out
#SBATCH --error=%j_cifar10_e16_T.err

hostname

nvidia-smi

python  -u train.py -p train -c config/adv_cifar10_e16.json