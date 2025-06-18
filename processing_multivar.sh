#!/bin/bash
#SBATCH --job-name="RADAR-SVK"
#SBATCH --output=radar_proc.out
#SBATCH --partition=long
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

#SBATCH --account=p709-24-2

srun /home/ppavlik/miniconda3/envs/nowcasting/bin/python /home/ppavlik/repos/radar-dataset-preprocess/processing_multivar.py