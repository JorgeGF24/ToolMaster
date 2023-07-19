# Simple test script to test the trainer on a GPT-2 small model

import os
from transformers import Trainer, TrainingArguments
import torch.nn as nn

import beartype
from beartype.typing import Callable, Any, TypeVar, Dict, Union, Optional, Sequence, Tuple


from transformers import GPT2Tokenizer, GPT2LMHeadModel


# Load the model
cache_dir = "/vol/bitbucket/jg2619/toolformer/cache/"
model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=cache_dir)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=cache_dir)

# Load the dataset
from datasets import load_dataset
dataset = load_dataset("/vol/bitbucket/jg2619/data/augmented_standard/", split="train", data_files="")
eval_dataset = load_dataset("/vol/bitbucket/jg2619/data/augmented_standard/", split="test")

# Train the model
training_args = TrainingArguments(
    output_dir="./results", #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=1, # number of training epochs
    per_device_train_batch_size=1, # batch size for training
    per_device_eval_batch_size=1,  # batch size for evaluation
    eval_steps = 100, # Number of update steps between two evaluations.
    save_steps=100, # after # steps model is saved
    warmup_steps=500,# number of warmup steps for learning rate scheduler
)

trainer = Trainer(
    model=model, # the instantiated 🤗 Transformers model to be trained
    args=training_args, # training arguments, defined above
    train_dataset=dataset, # training dataset
    #eval_dataset=dataset # evaluation dataset
)

print(trainer.use_apex)

