import torch

from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from argparse import ArgumentParser


class GPT2LM_fine_tuning():

    def __init__(self,  model_name_path: str) -> None:

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_name_path)

    def load_dataset(self, file_path: str, block_size: int):
        dataset = TextDataset(
            tokenizer = self.tokenizer,
            file_path = file_path,
            block_size = block_size,
        )
        return dataset

    def load_data_collator(self, mlm = False):
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=mlm,
        )
        return data_collator


    def train(self,
            train_path: str,
            eval_path: str,
            output_dir: str,
            block_size: int,
            batch_size: int,
            n_epochs: int,):
 
        train_dataset = self.load_dataset(block_size=block_size, file_path=train_path)
        eval_dataset = self.load_dataset(block_size=block_size, file_path=eval_path)
        data_collator = self.load_data_collator()

        self.tokenizer.save_pretrained(output_dir)
        self.model.save_pretrained(output_dir)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(device)

        training_args = TrainingArguments(
                output_dir=output_dir,
                overwrite_output_dir=True,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=n_epochs,
                save_strategy='epoch',
                evaluation_strategy='epoch',
                save_total_limit=1,
                do_train=True,
                do_eval=True,
            )

        trainer = Trainer(
                model=self.model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset
        )
            
        trainer.train()
        trainer.save_model()


def setup_train():


    parser = ArgumentParser(description='train.py script runs tAMPer for training.')

    parser.add_argument('--out', type=str, required=True,
                        help='output directory for writing the results')

    parser.add_argument('--train', type=str, required=True,
                        help='training text file path')

    parser.add_argument('--eval', type=str, required=True,
                        help='evaluation text file path')
    
    parser.add_argument('--epochs', type=int, required=True,
                        help='Max number of epochs for training')
    
    parser.add_argument('--batch_size', type=int, required=True)
        
    parser.add_argument('--block_size', type=int, required=True,
                        help='Block size of text for language modeling')
    

    args = parser.parse_args()

    gpt2 = GPT2LM_fine_tuning(model_name_path='gpt2')

    gpt2.train(
        train_path=args.train,
        eval_path=args.eval,
        output_dir=args.out,
        block_size=args.block_size,
        batch_size=args.batch_size,
        n_epochs=args.epochs
    )
