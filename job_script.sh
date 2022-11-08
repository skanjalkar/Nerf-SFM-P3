#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=6G

module load cuda11.1/toolkit
python3 ./Phase2/Wrapper.py
