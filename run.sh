#!/bin/bash -l 
#PBS -joe
which conda
module list
cd ~/ML/rva
python main.py
