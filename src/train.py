import torch

from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer

from argparse import ArgumentParser


class LLM_fine_tuning():

    def __init__(self,
                model_name_path: str,
                cache_dir: str,
                use_bnb: bool = True,) -> None:

        if use_bnb:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            use_cache=False,
            device_map="auto",
            cache_dir=cache_dir)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_path,
            trust_remote_code=True,
            cache_dir=cache_dir)

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

        lora_alpha = 16
        lora_dropout = 0.1
        lora_r = 64

        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM",
        )

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
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            data_collator=data_collator,
            dataset_text_field="text",
            tokenizer=self.tokenizer,
            args=training_args,
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

    gpt2 = LLM_fine_tuning(model_name_path='gpt2')

    gpt2.train(
        train_path=args.train,
        eval_path=args.eval,
        output_dir=args.out,
        block_size=args.block_size,
        batch_size=args.batch_size,
        n_epochs=args.epochs
    )


if __name__ == "__main__":
    setup_train()

