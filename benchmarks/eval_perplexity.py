from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader

import pandas as pd

from functools import partial


DEVICE = "cuda"

long_tensor = partial(torch.tensor, dtype=torch.long, device=DEVICE)


def perplexity(preds, labs):
    # Dataset is a tuple of predictions and labels

    total_perplexity = 0
    examples = 0

    for pred, lab in zip(preds, labs):
        examples += 1
        loss_fct = torch.nn.functional.cross_entropy(pred.squeeze(), lab.squeeze(), reduction='sum')

        total_perplexity += torch.exp(loss_fct)

    return {"perplexity": total_perplexity}


cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache"

def trained_models(name):
    return "/vol/bitbucket/jg2619/augmenting_llms/model_training/models/" + name

path = "EleutherAI/gpt-j-6B"

model = AutoModelForCausalLM.from_pretrained(
    path,
    revision="float16",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,          # CANT HANDLE DEEPSPEED ZERO 3
    cache_dir=cache_dir 
).cuda()

model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/gpt-j-6B",
    cache_dir=cache_dir 
)

tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = pd.read_csv("/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/unprocessed/segment_ccnet_unprocessed/1000_examples_not_in_training.csv")
# Dataloader:

processed_texts = 0

total_perplexity = 0

with torch.no_grad():
    while processed_texts < len(dataset):
        batch = tokenizer(dataset.iloc[processed_texts:processed_texts+2]["text"].to_list(), return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")

        output = model(**batch)
        processed_texts += 2

        total_perplexity += perplexity(output.logits[:,:-1], batch["input_ids"][:, 1:])["perplexity"]

        print(f"Perplexity: {total_perplexity / processed_texts}")

print(f"Perplexity: {total_perplexity / len(dataset)}")