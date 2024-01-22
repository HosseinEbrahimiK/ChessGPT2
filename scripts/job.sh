#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=70000M
#SBATCH --time=0-2:00

#command to run
# export split

source /home/hebz/envs/GPT2/bin/activate

cd /home/hebz/projects/def-ibirol/hebz/ChessGPT2/src/

python3 datasets.py