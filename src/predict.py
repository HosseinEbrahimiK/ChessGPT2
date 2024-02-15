from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from loguru import logger
from argparse import ArgumentParser


def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model

def load_tokenizer(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def play(model_path: str,
        tokenizer_path: str,
        moves: str,
        max_n_tokens: int):
    
    lora_config = LoraConfig.from_pretrained(model_path)
    model = get_peft_model(model, lora_config)
    tokenizer = load_tokenizer(tokenizer_path)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ids = tokenizer.encode(moves, return_tensors='pt').input_ids
    
    outputs = model.generate(
        input_ids=ids,
        max_new_tokens=max_n_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=0.9)

    logger.info(tokenizer.batch_decode(outputs.detach().cpu().numpy()))


if __name__ == "__main__":
    
    parser = ArgumentParser(description='predict.py script for generating chess moves.')

    tokenizer_path = '/home/hebz/projects/def-ibirol/hebz/ChessGPT2/chkpnts'
    model_path = '/home/hebz/projects/def-ibirol/hebz/ChessGPT2/chkpnts/checkpoint-77450'

    play(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        moves='1. e4',
        max_length=100
    )
    