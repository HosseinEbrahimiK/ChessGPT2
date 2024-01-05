# from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
# from transformers import TextDataset, DataCollatorForLanguageModeling
# from transformers import Trainer, TrainingArguments
import chess.pgn
import re


class LichessData:
    def __init__(self) -> None:
        pass


if __name__ == "__main__":
    pgn = open("/home/hebz/scratch/chessGPT/lichess_Nov_2023/lichess_db_standard_rated_2023-11.pgn")
    game = chess.pgn.read_game(pgn)
    games = []

    while game:
    
        game_str = str(game)
        moves = re.sub(r'\[.*?\]|\{.*?\}', '', game_str)
        moves = re.sub(r'\b\d+\s*\.\.\.\s*\b', '', moves)
        moves = re.sub(r'\s{2,}', ' ', moves)[1:]
        games.append(moves)

        game = chess.pgn.read_game(pgn)

    print(len(games))

