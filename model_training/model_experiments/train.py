from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel

import torch
from datasets import load_dataset
import os

from functools import partial
from torch.nn.utils.rnn import pad_sequence

from transformers import Trainer, TrainingArguments
import torch.nn as nn

import ast
from accelerate.utils import DummyOptim, DummyScheduler

pad_sequence = partial(pad_sequence, batch_first=True)

GPTJ_LAYERS = {'Embedding Layer': ['transformer.wte.weight'],
 'Layer 0': ['transformer.h.0.ln_1.weight',
  'transformer.h.0.ln_1.bias',
  'transformer.h.0.attn.k_proj.weight',
  'transformer.h.0.attn.v_proj.weight',
  'transformer.h.0.attn.q_proj.weight',
  'transformer.h.0.attn.out_proj.weight',
  'transformer.h.0.mlp.fc_in.weight',
  'transformer.h.0.mlp.fc_in.bias',
  'transformer.h.0.mlp.fc_out.weight',
  'transformer.h.0.mlp.fc_out.bias'],
 'Layer 1': ['transformer.h.1.ln_1.weight',
  'transformer.h.1.ln_1.bias',
  'transformer.h.1.attn.k_proj.weight',
  'transformer.h.1.attn.v_proj.weight',
  'transformer.h.1.attn.q_proj.weight',
  'transformer.h.1.attn.out_proj.weight',
  'transformer.h.1.mlp.fc_in.weight',
  'transformer.h.1.mlp.fc_in.bias',
  'transformer.h.1.mlp.fc_out.weight',
  'transformer.h.1.mlp.fc_out.bias'],
 'Layer 2': ['transformer.h.2.ln_1.weight',
  'transformer.h.2.ln_1.bias',
  'transformer.h.2.attn.k_proj.weight',
  'transformer.h.2.attn.v_proj.weight',
  'transformer.h.2.attn.q_proj.weight',
  'transformer.h.2.attn.out_proj.weight',
  'transformer.h.2.mlp.fc_in.weight',
  'transformer.h.2.mlp.fc_in.bias',
  'transformer.h.2.mlp.fc_out.weight',
  'transformer.h.2.mlp.fc_out.bias'],
 'Layer 3': ['transformer.h.3.ln_1.weight',
  'transformer.h.3.ln_1.bias',
  'transformer.h.3.attn.k_proj.weight',
  'transformer.h.3.attn.v_proj.weight',
  'transformer.h.3.attn.q_proj.weight',
  'transformer.h.3.attn.out_proj.weight',
  'transformer.h.3.mlp.fc_in.weight',
  'transformer.h.3.mlp.fc_in.bias',
  'transformer.h.3.mlp.fc_out.weight',
  'transformer.h.3.mlp.fc_out.bias'],
 'Layer 4': ['transformer.h.4.ln_1.weight',
  'transformer.h.4.ln_1.bias',
  'transformer.h.4.attn.k_proj.weight',
  'transformer.h.4.attn.v_proj.weight',
  'transformer.h.4.attn.q_proj.weight',
  'transformer.h.4.attn.out_proj.weight',
  'transformer.h.4.mlp.fc_in.weight',
  'transformer.h.4.mlp.fc_in.bias',
  'transformer.h.4.mlp.fc_out.weight',
  'transformer.h.4.mlp.fc_out.bias'],
 'Layer 5': ['transformer.h.5.ln_1.weight',
  'transformer.h.5.ln_1.bias',
  'transformer.h.5.attn.k_proj.weight',
  'transformer.h.5.attn.v_proj.weight',
  'transformer.h.5.attn.q_proj.weight',
  'transformer.h.5.attn.out_proj.weight',
  'transformer.h.5.mlp.fc_in.weight',
  'transformer.h.5.mlp.fc_in.bias',
  'transformer.h.5.mlp.fc_out.weight',
  'transformer.h.5.mlp.fc_out.bias'],
 'Layer 6': ['transformer.h.6.ln_1.weight',
  'transformer.h.6.ln_1.bias',
  'transformer.h.6.attn.k_proj.weight',
  'transformer.h.6.attn.v_proj.weight',
  'transformer.h.6.attn.q_proj.weight',
  'transformer.h.6.attn.out_proj.weight',
  'transformer.h.6.mlp.fc_in.weight',
  'transformer.h.6.mlp.fc_in.bias',
  'transformer.h.6.mlp.fc_out.weight',
  'transformer.h.6.mlp.fc_out.bias'],
 'Layer 7': ['transformer.h.7.ln_1.weight',
  'transformer.h.7.ln_1.bias',
  'transformer.h.7.attn.k_proj.weight',
  'transformer.h.7.attn.v_proj.weight',
  'transformer.h.7.attn.q_proj.weight',
  'transformer.h.7.attn.out_proj.weight',
  'transformer.h.7.mlp.fc_in.weight',
  'transformer.h.7.mlp.fc_in.bias',
  'transformer.h.7.mlp.fc_out.weight',
  'transformer.h.7.mlp.fc_out.bias'],
 'Layer 8': ['transformer.h.8.ln_1.weight',
  'transformer.h.8.ln_1.bias',
  'transformer.h.8.attn.k_proj.weight',
  'transformer.h.8.attn.v_proj.weight',
  'transformer.h.8.attn.q_proj.weight',
  'transformer.h.8.attn.out_proj.weight',
  'transformer.h.8.mlp.fc_in.weight',
  'transformer.h.8.mlp.fc_in.bias',
  'transformer.h.8.mlp.fc_out.weight',
  'transformer.h.8.mlp.fc_out.bias'],
 'Layer 9': ['transformer.h.9.ln_1.weight',
  'transformer.h.9.ln_1.bias',
  'transformer.h.9.attn.k_proj.weight',
  'transformer.h.9.attn.v_proj.weight',
  'transformer.h.9.attn.q_proj.weight',
  'transformer.h.9.attn.out_proj.weight',
  'transformer.h.9.mlp.fc_in.weight',
  'transformer.h.9.mlp.fc_in.bias',
  'transformer.h.9.mlp.fc_out.weight',
  'transformer.h.9.mlp.fc_out.bias'],
 'Layer 10': ['transformer.h.10.ln_1.weight',
  'transformer.h.10.ln_1.bias',
  'transformer.h.10.attn.k_proj.weight',
  'transformer.h.10.attn.v_proj.weight',
  'transformer.h.10.attn.q_proj.weight',
  'transformer.h.10.attn.out_proj.weight',
  'transformer.h.10.mlp.fc_in.weight',
  'transformer.h.10.mlp.fc_in.bias',
  'transformer.h.10.mlp.fc_out.weight',
  'transformer.h.10.mlp.fc_out.bias'],
 'Layer 11': ['transformer.h.11.ln_1.weight',
  'transformer.h.11.ln_1.bias',
  'transformer.h.11.attn.k_proj.weight',
  'transformer.h.11.attn.v_proj.weight',
  'transformer.h.11.attn.q_proj.weight',
  'transformer.h.11.attn.out_proj.weight',
  'transformer.h.11.mlp.fc_in.weight',
  'transformer.h.11.mlp.fc_in.bias',
  'transformer.h.11.mlp.fc_out.weight',
  'transformer.h.11.mlp.fc_out.bias'],
 'Layer 12': ['transformer.h.12.ln_1.weight',
  'transformer.h.12.ln_1.bias',
  'transformer.h.12.attn.k_proj.weight',
  'transformer.h.12.attn.v_proj.weight',
  'transformer.h.12.attn.q_proj.weight',
  'transformer.h.12.attn.out_proj.weight',
  'transformer.h.12.mlp.fc_in.weight',
  'transformer.h.12.mlp.fc_in.bias',
  'transformer.h.12.mlp.fc_out.weight',
  'transformer.h.12.mlp.fc_out.bias'],
 'Layer 13': ['transformer.h.13.ln_1.weight',
  'transformer.h.13.ln_1.bias',
  'transformer.h.13.attn.k_proj.weight',
  'transformer.h.13.attn.v_proj.weight',
  'transformer.h.13.attn.q_proj.weight',
  'transformer.h.13.attn.out_proj.weight',
  'transformer.h.13.mlp.fc_in.weight',
  'transformer.h.13.mlp.fc_in.bias',
  'transformer.h.13.mlp.fc_out.weight',
  'transformer.h.13.mlp.fc_out.bias'],
 'Layer 14': ['transformer.h.14.ln_1.weight',
  'transformer.h.14.ln_1.bias',
  'transformer.h.14.attn.k_proj.weight',
  'transformer.h.14.attn.v_proj.weight',
  'transformer.h.14.attn.q_proj.weight',
  'transformer.h.14.attn.out_proj.weight',
  'transformer.h.14.mlp.fc_in.weight',
  'transformer.h.14.mlp.fc_in.bias',
  'transformer.h.14.mlp.fc_out.weight',
  'transformer.h.14.mlp.fc_out.bias'],
 'Layer 15': ['transformer.h.15.ln_1.weight',
  'transformer.h.15.ln_1.bias',
  'transformer.h.15.attn.k_proj.weight',
  'transformer.h.15.attn.v_proj.weight',
  'transformer.h.15.attn.q_proj.weight',
  'transformer.h.15.attn.out_proj.weight',
  'transformer.h.15.mlp.fc_in.weight',
  'transformer.h.15.mlp.fc_in.bias',
  'transformer.h.15.mlp.fc_out.weight',
  'transformer.h.15.mlp.fc_out.bias'],
 'Layer 16': ['transformer.h.16.ln_1.weight',
  'transformer.h.16.ln_1.bias',
  'transformer.h.16.attn.k_proj.weight',
  'transformer.h.16.attn.v_proj.weight',
  'transformer.h.16.attn.q_proj.weight',
  'transformer.h.16.attn.out_proj.weight',
  'transformer.h.16.mlp.fc_in.weight',
  'transformer.h.16.mlp.fc_in.bias',
  'transformer.h.16.mlp.fc_out.weight',
  'transformer.h.16.mlp.fc_out.bias'],
 'Layer 17': ['transformer.h.17.ln_1.weight',
  'transformer.h.17.ln_1.bias',
  'transformer.h.17.attn.k_proj.weight',
  'transformer.h.17.attn.v_proj.weight',
  'transformer.h.17.attn.q_proj.weight',
  'transformer.h.17.attn.out_proj.weight',
  'transformer.h.17.mlp.fc_in.weight',
  'transformer.h.17.mlp.fc_in.bias',
  'transformer.h.17.mlp.fc_out.weight',
  'transformer.h.17.mlp.fc_out.bias'],
 'Layer 18': ['transformer.h.18.ln_1.weight',
  'transformer.h.18.ln_1.bias',
  'transformer.h.18.attn.k_proj.weight',
  'transformer.h.18.attn.v_proj.weight',
  'transformer.h.18.attn.q_proj.weight',
  'transformer.h.18.attn.out_proj.weight',
  'transformer.h.18.mlp.fc_in.weight',
  'transformer.h.18.mlp.fc_in.bias',
  'transformer.h.18.mlp.fc_out.weight',
  'transformer.h.18.mlp.fc_out.bias'],
 'Layer 19': ['transformer.h.19.ln_1.weight',
  'transformer.h.19.ln_1.bias',
  'transformer.h.19.attn.k_proj.weight',
  'transformer.h.19.attn.v_proj.weight',
  'transformer.h.19.attn.q_proj.weight',
  'transformer.h.19.attn.out_proj.weight',
  'transformer.h.19.mlp.fc_in.weight',
  'transformer.h.19.mlp.fc_in.bias',
  'transformer.h.19.mlp.fc_out.weight',
  'transformer.h.19.mlp.fc_out.bias'],
 'Layer 20': ['transformer.h.20.ln_1.weight',
  'transformer.h.20.ln_1.bias',
  'transformer.h.20.attn.k_proj.weight',
  'transformer.h.20.attn.v_proj.weight',
  'transformer.h.20.attn.q_proj.weight',
  'transformer.h.20.attn.out_proj.weight',
  'transformer.h.20.mlp.fc_in.weight',
  'transformer.h.20.mlp.fc_in.bias',
  'transformer.h.20.mlp.fc_out.weight',
  'transformer.h.20.mlp.fc_out.bias'],
 'Layer 21': ['transformer.h.21.ln_1.weight',
  'transformer.h.21.ln_1.bias',
  'transformer.h.21.attn.k_proj.weight',
  'transformer.h.21.attn.v_proj.weight',
  'transformer.h.21.attn.q_proj.weight',
  'transformer.h.21.attn.out_proj.weight',
  'transformer.h.21.mlp.fc_in.weight',
  'transformer.h.21.mlp.fc_in.bias',
  'transformer.h.21.mlp.fc_out.weight',
  'transformer.h.21.mlp.fc_out.bias'],
 'Layer 22': ['transformer.h.22.ln_1.weight',
  'transformer.h.22.ln_1.bias',
  'transformer.h.22.attn.k_proj.weight',
  'transformer.h.22.attn.v_proj.weight',
  'transformer.h.22.attn.q_proj.weight',
  'transformer.h.22.attn.out_proj.weight',
  'transformer.h.22.mlp.fc_in.weight',
  'transformer.h.22.mlp.fc_in.bias',
  'transformer.h.22.mlp.fc_out.weight',
  'transformer.h.22.mlp.fc_out.bias'],
 'Layer 23': ['transformer.h.23.ln_1.weight',
  'transformer.h.23.ln_1.bias',
  'transformer.h.23.attn.k_proj.weight',
  'transformer.h.23.attn.v_proj.weight',
  'transformer.h.23.attn.q_proj.weight',
  'transformer.h.23.attn.out_proj.weight',
  'transformer.h.23.mlp.fc_in.weight',
  'transformer.h.23.mlp.fc_in.bias',
  'transformer.h.23.mlp.fc_out.weight',
  'transformer.h.23.mlp.fc_out.bias'],
 'Layer 24': ['transformer.h.24.ln_1.weight',
  'transformer.h.24.ln_1.bias',
  'transformer.h.24.attn.k_proj.weight',
  'transformer.h.24.attn.v_proj.weight',
  'transformer.h.24.attn.q_proj.weight',
  'transformer.h.24.attn.out_proj.weight',
  'transformer.h.24.mlp.fc_in.weight',
  'transformer.h.24.mlp.fc_in.bias',
  'transformer.h.24.mlp.fc_out.weight',
  'transformer.h.24.mlp.fc_out.bias'],
 'Layer 25': ['transformer.h.25.ln_1.weight',
  'transformer.h.25.ln_1.bias',
  'transformer.h.25.attn.k_proj.weight',
  'transformer.h.25.attn.v_proj.weight',
  'transformer.h.25.attn.q_proj.weight',
  'transformer.h.25.attn.out_proj.weight',
  'transformer.h.25.mlp.fc_in.weight',
  'transformer.h.25.mlp.fc_in.bias',
  'transformer.h.25.mlp.fc_out.weight',
  'transformer.h.25.mlp.fc_out.bias'],
 'Layer 26': ['transformer.h.26.ln_1.weight',
  'transformer.h.26.ln_1.bias',
  'transformer.h.26.attn.k_proj.weight',
  'transformer.h.26.attn.v_proj.weight',
  'transformer.h.26.attn.q_proj.weight',
  'transformer.h.26.attn.out_proj.weight',
  'transformer.h.26.mlp.fc_in.weight',
  'transformer.h.26.mlp.fc_in.bias',
  'transformer.h.26.mlp.fc_out.weight',
  'transformer.h.26.mlp.fc_out.bias'],
 'Layer 27': ['transformer.h.27.ln_1.weight',
  'transformer.h.27.ln_1.bias',
  'transformer.h.27.attn.k_proj.weight',
  'transformer.h.27.attn.v_proj.weight',
  'transformer.h.27.attn.q_proj.weight',
  'transformer.h.27.attn.out_proj.weight',
  'transformer.h.27.mlp.fc_in.weight',
  'transformer.h.27.mlp.fc_in.bias',
  'transformer.h.27.mlp.fc_out.weight',
  'transformer.h.27.mlp.fc_out.bias'],
 'Final Layer Norm': ['transformer.ln_f.weight', 'transformer.ln_f.bias'],
 'LM Head': ['lm_head.weight', 'lm_head.bias']}


cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache/"
dataset_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/augmented_standard/"
data_files = os.listdir("/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/augmented_prompttrick/")
len_data_files = len(data_files)

gpt2 = False

if gpt2:
    model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=cache_dir)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=cache_dir)
else:
    model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B",
            revision="float16",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,          # CANT HANDLE DEEPSPEED ZERO 3
            cache_dir=cache_dir 
        ).cuda()
    tokenizer = AutoTokenizer.from_pretrained("/vol/bitbucket/jg2619/augmenting_llms/model_training/models/tokenizer", truncate=True, max_length=270, cache_dir=cache_dir)

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

FROZEN_LAYERS = []
for i in range(0, 23):
    FROZEN_LAYERS += GPTJ_LAYERS[f"Layer {i}"]

# Freeze some layers in the architecture
for name, param in model.named_parameters():
    if name in FROZEN_LAYERS:
        param.requires_grad = False


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
    data = [torch.tensor(item[0]).long() for item in batch]
    mask = [torch.tensor(item[1]).int() for item in batch]
    data = pad_sequence(data, batch_first=True).to(DEVICE)
    mask = pad_sequence(mask, batch_first=True).to(DEVICE)
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
    dataloader_pin_memory=False,
    deepspeed="/vol/bitbucket/jg2619/augmenting_llms/model_training/model_experiments/ds_conf.json",
    gradient_accumulation_steps=16,
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