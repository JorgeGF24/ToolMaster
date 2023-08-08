from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPT2LMHeadModel

import torch
from datasets import load_dataset
import os

from functools import partial
from torch.nn.utils.rnn import pad_sequence

from transformers import Trainer, TrainingArguments
import torch.nn as nn

from model_training.model_experiments.GPTJ_layers import GPTJ_LAYERS

import ast

pad_sequence = partial(pad_sequence, batch_first=True)



cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache/"
dataset_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/augmented_standard/"
data_files = os.listdir("/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/augmented_prompttrick/")
len_data_files = len(data_files)

MODEL_NAME = "GPTJ"

if MODEL_NAME == "GPT2":
    model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=cache_dir)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=cache_dir)
elif MODEL_NAME == "GPTJ":
    model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B",
            revision="float16",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,          # CANT HANDLE DEEPSPEED ZERO 3
            cache_dir=cache_dir 
        ).cuda()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", truncate=True, max_length=270, cache_dir=cache_dir)

TOOL_START_TOKEN = "<TOOL>"
TOOL_END_TOKEN = "</TOOL>" 

if MODEL_NAME == "GPTJ":
    TOOL_START_TOKEN = " " + TOOL_START_TOKEN


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
print(f"Len tokenizer: {len(tokenizer)}")

DEVICE = model.device

FROZEN_LAYERS = []
TRAINED_LAYERS = []
for i in range(0, 23):
    FROZEN_LAYERS += GPTJ_LAYERS[f"Layer {i}"]

# Freeze some layers in the architecture
for name, param in model.named_parameters():
    if name in FROZEN_LAYERS:
        param.requires_grad = False
    else:
        TRAINED_LAYERS.append(name)

available_tools_prompt = "The following tools are available: "

tool_name_alternatives = {
    "Calculator":["Calculator","calculator","CALCULATOR"
                  "Calculate","calculate","CALCULATE",
                  "Operate","operate","OPERATE",
                  "Calc","calc","CALC",
                  "WolframAlpha","wolframAlpha","WOLFRAMALPHA","Wolframalpha",
                  "Add","add","ADD",
                  "Subtract","subtract","SUBTRACT",
                  "Multiply","multiply","MULTIPLY",
                  "Divide","divide","DIVIDE",
                  "Arithmetic","arithmetic","ARITHMETIC",],
    "WikiSearch":["WikiSearch","wikisearch","WIKISEARCH","Wiki-search","wiki-Search","WIKI-SEARCH","Wiki_search","Wiki_Search","WIKI_SEARCH",
                  "Search","search","SEARCH",
                  "Wiki","wiki","WIKI",
                  "WikiPedia","wikipedia","WIKIPEDIA","Wikipedia",
                  "InternetSearch","internetSearch","INTERNETSEARCH","Internet-search","internet-search","INTERNET-SEARCH","Internet_search","Internet_Search","INTERNET_SEARCH",
                  "Google","google","GOOGLE",
                  "Browser","browser","BROWSER",
                  "Knowledge-base","knowledge-base","KNOWLEDGE-BASE","Knowledge_base","Knowledge_Base","KNOWLEDGE_BASE",]
}


# Dataset classes and collate functions
class ToolDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        self.texts = dataset["tokenized_tool_text"]
        self.masks = dataset["method_A_train_mask"]

        """
        self.start_texts = dataset["start_tokenized_tool_text"]
        self.end_texts = dataset["end_tokenized_tool_text"]
        self.start_masks = dataset["start_method_A_train_mask"]
        self.end_masks = dataset["end_method_A_train_mask"]
        self.tool_names = dataset["tool_name"]"""
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.masks)
    
    def __getitem__(self, idx):
        """
        start_text = self.start_texts[idx]
        end_text = self.end_texts[idx]
        start_mask = self.start_mask[idx]
        end_mask = self.end_mask[idx]"""



        return ast.literal_eval(self.texts[idx]), ast.literal_eval(self.masks[idx])

DEBUG = True

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

    # On init, init super class:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        tokenized_data, mask = inputs

        mask = mask[:,1:]  #Prediction starts from second token


        outputs = model(**{"input_ids":tokenized_data[:,:-1]}, use_cache=False)


        # For debbuging purposes, print columns with 1. Decoded token, 2. mask value:
        # The columns should be well padded with spaces, so that the mask values are aligned
        # Example:
        # Data      //    mask:
        # <TOOL>    //      1
        # <TOOL>    //      1
        # You       //      0
        if DEBUG:
            # SIZES OF INPUTS:
            print(f"Tokenized data size: {tokenized_data.shape}")
            print("Data      //    mask:")
            logits = outputs.logits
            for i, sentence in enumerate(tokenized_data[:5]):
                for j, token in enumerate(sentence):
                    print(f"{tokenizer.decode(token):<10} // {mask[i][min(mask.shape[1]-1,j)].item():>6}")
                    print(f"Predicted token: {tokenizer.decode(logits[i][min(logits.shape[1]-1,j)].argmax().item())}")
        
        mask = mask.view(-1).bool()
        logits = outputs.logits
        logits = logits.view(-1, logits.shape[-1])[mask]
        tokenized_data = tokenized_data[:,1:].view(-1)[mask]
        loss = torch.nn.functional.cross_entropy(logits, tokenized_data)

        if DEBUG:
            print(f"Loss:{loss.item()}")
            
            for i, logit in enumerate(logits):
                print(f"Predicted token: {tokenizer.decode(logit.argmax().item())}")
                print(f"{tokenizer.decode(tokenized_data[i]):<10} // {mask[i].item():>6}")


        return (loss, outputs) if return_outputs else loss
    



# Train the model
training_args = TrainingArguments(
    output_dir="./results", # The output directory
    overwrite_output_dir=True, # overwrite the content of the output directory
    num_train_epochs=1, # number of training epochs
    per_device_train_batch_size=1, # batch size for training
    per_device_eval_batch_size=1,  # batch size for evaluation
    eval_steps = 10, # Number of update steps between two evaluations.
    save_steps=1000, # after # steps model is saved
    warmup_steps=500,# number of warmup steps for learning rate scheduler
    dataloader_pin_memory=False,
    do_train=True,
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

print("END OF SETUP")

def main():

    print("GONNA TRAIN")
    trainer.train()

    # Print cuda memory summary
    print(torch.cuda.memory_summary())

    model.save_pretrained("/vol/bitbucket/jg2619/augmenting_llms/model_training/models/GPT2")


if __name__ == "__main__":
    main()