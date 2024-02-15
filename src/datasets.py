# from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
# from transformers import TextDataset, DataCollatorForLanguageModeling
# from transformers import Trainer, TrainingArguments
import chess.pgn
import re
from loguru import logger


class LichessDataset:
    def __init__(self) -> None:
        self.games = []


def read_games(file_address: str, TRAIN_NUM=500000, VAL_NUM=100000) -> None:

    logger.info("Started reading .pgn file")
    pgn = open(file_address)
    num_games = 0
    train = ''
    val = ''
    game = chess.pgn.read_game(pgn)

    while num_games < TRAIN_NUM + VAL_NUM:
        
        if num_games % 10000 == 0:
            logger.info(f'{num_games} has been read!!')

        game_str = str(game)
        moves = re.sub(r'\[.*?\]|\{.*?\}', '', game_str)
        moves = re.sub(r'\b\d+\s*\.\.\.\s*\b', '', moves)
        moves = re.sub(r'\$\S+', '', moves)
        moves = re.sub(r'\s{2,}', ' ', moves)

        if num_games < TRAIN_NUM:
            train += moves
        else:
            val += moves

        num_games += 1

        game = chess.pgn.read_game(pgn)

    with open("/home/hebz/scratch/chessGPT/100000_games/train.txt", "w") as f:
        f.write(train)

    with open("/home/hebz/scratch/chessGPT/100000_games/val.txt", "w") as f:
        f.write(val)
    
    return

if __name__ == "__main__":
    read_games('/home/hebz/scratch/chessGPT/lichess_Nov_2023/lichess_db_standard_rated_2023-11.pgn', TRAIN_NUM=100000, VAL_NUM=10000)