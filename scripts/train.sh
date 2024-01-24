#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10000M
#SBATCH --gpus-per-node=v100l:2
#SBATCH --time=0-25:00
#SBATCH --job-name GPT2
#SBATCH --output /home/hebz/projects/def-ibirol/hebz/ChessGPT2/logs/log_GPT2_50_epochs.out

# export split

source /home/hebz/envs/GPT2/bin/activate

cd /home/hebz/projects/def-ibirol/hebz/ChessGPT2/src/

python3 train.py --out /home/hebz/projects/def-ibirol/hebz/ChessGPT2/chkpnts \
    --train /home/hebz/scratch/chessGPT/100000_games/train.txt \
    --eval /home/hebz/scratch/chessGPT/100000_games/val.txt \
    --block_size 1024 \
    --batch_size 8 \
    --epochs 50