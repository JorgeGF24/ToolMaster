from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel
from transformers import DataCollatorForWholeWordMask

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Value, Features
import os

from functools import partial
from torch.nn.utils.rnn import pad_sequence

from transformers import Trainer, TrainingArguments
import torch.nn as nn

import ast

import beartype
from beartype.typing import Callable, Any, TypeVar, Dict, Union, Optional, Sequence, Tuple


pad_sequence = partial(pad_sequence, batch_first=True)


cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache/"
dataset_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/augmented_standard/"
data_files = os.listdir("/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/augmented_prompttrick/")
len_data_files = len(data_files)



model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=cache_dir)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=cache_dir)

TOOL_START_TOKEN = "<TOOL>"
TOOL_END_TOKEN = "</TOOL>" 

tokenizer.add_tokens([TOOL_START_TOKEN, TOOL_END_TOKEN, "[PAD]"])
tokenizer.pad_token = "[PAD]"
"""
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", padding=True, truncate=True, max_length=270)
tokenizer.add_tokens('[PAD]')

model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B",
        revision="float16",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
).cuda()"""

model.resize_token_embeddings(len(tokenizer))

DEVICE = model.device

# Dataset classes and collate functions
class ToolDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        self.texts = dataset["tokenized_tool_text"]
        self.masks = dataset["method_A_train_mask"]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.masks)
    
    def __getitem__(self, idx):
        text = ast.literal_eval(self.texts[idx])
        mask = ast.literal_eval(self.masks[idx])
        return text, mask


def collate_fn(batch):
    global DEVICE
    data = [torch.tensor(item[0], device=DEVICE).long() for item in batch]
    mask = [torch.tensor(item[1], device=DEVICE).int() for item in batch]
    data = pad_sequence(data, batch_first=True)
    mask = pad_sequence(mask, batch_first=True)
    return data, mask

# Arange of tensors from 0 to 2*3*4:
# >>> torch.arange(24).reshape(2,3,4)
#data_files=dataset_dir[len_data_files*0.8:]
#data_files=dataset_dir[:len_data_files*0.8]
train_dataset = load_dataset(dataset_dir + "train", split="train", cache_dir=cache_dir)
test_dataset = load_dataset(dataset_dir + "test", split="test", cache_dir=cache_dir)

train_dataset = ToolDataset({key: train_dataset[key] for key in ["tokenized_tool_text", "method_A_train_mask"]},tokenizer)
test_dataset = ToolDataset({key: test_dataset[key] for key in ["tokenized_tool_text", "method_A_train_mask"]}, tokenizer)

# Override the Trainer class to use our loss function

class MyTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        tokenized_data, mask = inputs
        mask = mask.view(-1).bool()
        print(tokenized_data)
        print(f"Tokenized data is: {tokenizer.decode(tokenized_data[0])}")
        outputs = model(**{"input_ids":tokenized_data}, use_cache=False)
        
        logits = outputs.logits
        logits = logits.view(-1, logits.shape[-1])[mask]
        tokenized_data = tokenized_data.view(-1)[mask]
        loss = torch.nn.functional.cross_entropy(logits, tokenized_data)
        return (loss, outputs) if return_outputs else loss


# Train the model
training_args = TrainingArguments(
    output_dir="./results", # The output directory
    overwrite_output_dir=True, # overwrite the content of the output directory
    num_train_epochs=1, # number of training epochs
    per_device_train_batch_size=1, # batch size for training
    per_device_eval_batch_size=1,  # batch size for evaluation
    eval_steps = 100, # Number of update steps between two evaluations.
    save_steps=1000, # after # steps model is saved
    warmup_steps=500,# number of warmup steps for learning rate scheduler
)

trainer = MyTrainer(
    model=model, # the instantiated 🤗 Transformers model to be trained
    args=training_args, # training arguments, defined above
    train_dataset=train_dataset, # training dataset
    eval_dataset=test_dataset, # evaluation dataset
    data_collator = collate_fn,
)

# TODO
#CHECKPOINTING
#STATS

trainer.train()

model.save_pretrained("/vol/bitbucket/jg2619/augmenting_llms/model_training/models/GPT2")