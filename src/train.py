from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from argparse import ArgumentParser


class GPT2LM_fine_tuning():

    def __init__(self, file_path: str, model_name_path: str) -> None:

        self.train_path = file_path
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_name_path)

    def load_dataset(self, block_size: int):
        dataset = TextDataset(
            tokenizer = self.tokenizer,
            file_path = self.train_path,
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
            output_dir: str,
            block_size: int,
            batch_size: int,
            n_epochs: int,
            run_name: str):
 
        train_dataset = self.load_dataset(block_size=block_size)
        eval_dataset = self.load_dataset(block_size=block_size)
        data_collator = self.load_data_collator()

        self.tokenizer.save_pretrained(output_dir)
        self.model.save_pretrained(output_dir)

        training_args = TrainingArguments(
                output_dir=output_dir,
                overwrite_output_dir=True,
                per_device_train_batch_size=batch_size,
                num_train_epochs=n_epochs,
                save_strategy='epoch',
                save_total_limit=1,
                run_name=run_name
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
    # data
    parser.add_argument('-out', type=str, required=True,
                        help='output directory for writing the results')

    parser.add_argument('-tr', type=str, required=True,
                        help='training text file path')
    
    parser.add_argument('-epochs', type=int, required=True,
                        help='Max number of epochs for training')
    
    parser.add_argument('-epochs', type=int, required=True,
                        help='Max number of epochs for training')