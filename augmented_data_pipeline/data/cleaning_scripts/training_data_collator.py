# This script readies the data for training. It takes the raw data from the dataset and converts it into a format that can be used by the model.

from csv import DictWriter
import os
import torch
import re
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM

cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache"
tokenizer = AutoTokenizer.from_pretrained("/vol/bitbucket/jg2619/models/tokenizer", truncate=True, max_length=270, cache_dir=cache_dir)

TOOL_START_TOKEN = "<TOOL>"
TOOL_END_TOKEN = "</TOOL>" 


def mask_tokenize_data(
        dataset_dir,
        output_dir,
        tool_name:str
):
    global tokenizer, TOOL_START_TOKEN, TOOL_END_TOKEN

    output_fields = ["tokenized_tool_text", "token_type", "method_A_train_mask", "tool_name"]
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_list = [file for file in os.listdir(dataset_dir) if file.endswith('.csv')]

    for file in file_list:
        dataset = load_dataset(dataset_dir, split="train", data_files = file)
        data_iter = iter(dataset)

        # Create output file
        with open(f"{output_dir}/{file}", 'w') as f:
            writer = DictWriter(f, fieldnames=output_fields)
            writer.writeheader()

            # Iterate through the dataset and write to the output file
            data = next(data_iter, None)
            output = {}
            while data is not None:
                
                text = data["API_call_response_text"]
                tokenized_text = tokenizer(text, truncation=True, max_length=270).input_ids

                # Find index where tokenized_text matches the tool start token:
                index_start = tokenized_text.index(tokenizer.encode(TOOL_START_TOKEN)[0])
                index_arrow = tokenized_text[index_start:].index(tokenizer.encode("→")[0]) + index_start + 1
                index_end = tokenized_text[index_arrow:].index(tokenizer.encode(TOOL_END_TOKEN)[0]) + index_arrow + 1

                len_start = index_start + 1

                len_toolname = len(tokenizer.encode(tool_name))

                # Find number of ocurrences of →
                occurrences = len(re.findall(r'(\)\→)', text))

                if occurrences == 1:
                    # Create mask for method A
                    method_A_train_mask = torch.zeros(len(tokenized_text))
                    method_A_train_mask[len_start+len_toolname] += 1
                    output["method_A_train_mask"] = method_A_train_mask
                else:
                    print("More than one occurrence of →", flush=True)
                    print(text, flush=True)
                    raise Exception("More than one occurrence of →")

                # Create token type mask
                token_type = torch.zeros(len(tokenized_text))      # Data...
                token_type[len_start] += 1                         # <TOOL>
                token_type[len_start+1] += 1                       # Toolname
                token_type[len_start+1 + len_toolname] += 1        # (
                token_type[len_start+1 + len_toolname + 1] += 1    # args
                token_type[index_arrow - 1] += 1                   # )
                token_type[index_arrow] += 1                       # →
                token_type[index_arrow+1] += 1                     # response
                token_type[index_end] += 1                         # </TOOL>
                token_type[index_end+1] += 1                       # ...Data
                token_type = token_type.cumsum(dim=0)

                method_A_train_mask = (torch.isin(token_type, torch.tensor([0, 1, 2, 9])))

                output["tokenized_tool_text"] = tokenized_text
                output["token_type"] = token_type.tolist()
                output["method_A_train_mask"] = method_A_train_mask.float().tolist()
                output["tool_name"] = tool_name

                writer.writerow(output)

                data = next(data_iter, None)


if __name__ == "__main__":
    dataset_dir = "/vol/bitbucket/jg2619/data/augmented_standard/processed"
    output_dir = "/vol/bitbucket/jg2619/data/augmented_standard/test/train_ready/"
    mask_tokenize_data(dataset_dir, output_dir, "WikiSearch")



