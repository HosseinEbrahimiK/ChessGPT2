#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=50000M
#SBATCH --time=0-00:20

#command to run
# export split

source /home/hebz/envs/GPT2/bin/activate

cd /home/hebz/projects/def-ibirol/hebz/ChessGPT2/src/

python3 predict.py۲۲