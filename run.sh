#!/bin/bash -l 
#PBS -joe
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l walltime=00:02:00

module unload ml-toolkit-gpu/pytorch/1.0.0
source activate pytorch
which conda
module list
cd ~/ML/rva

python main.py
