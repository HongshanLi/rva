#!/bin/bash -l 
#PBS -joe
#PBS -q highmem
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l walltime=00:05:00

source activate pytorch
cd ~/ML/rva

python main.py
