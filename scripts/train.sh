#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10000M
#SBATCH --gres=gpu:1
#SBATCH --time=0-03:00
#SBATCH --job-name GPT2
#SBATCH --output /home/hebz/projects/def-ibirol/hebz/ChessGPT2/logs/log_GPT2.out

# export split

source /home/hebz/envs/chessGPT/bin/activate

cd /home/hebz/projects/def-ibirol/hebz/ChessGPT2/src/

python3 train.py --out /home/hebz/projects/def-ibirol/hebz/ChessGPT2/chkpnts \
    --train /home/hebz/scratch/chessGPT/500000_games/train.txt \
    --eval /home/hebz/scratch/chessGPT/500000_games/train.txt \
    --block_size 128 \
    --batch_size 8 \
    --epochs 10