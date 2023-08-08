# This script readies the data for training. It takes the raw data from the dataset and converts it into a format that can be used by the model.

from csv import DictWriter
import os
import torch
import re
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM

cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache"
tokenizer = AutoTokenizer.from_pretrained("/vol/bitbucket/jg2619/models/tokenizer", truncate=True, max_length=270, cache_dir=cache_dir)
tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2", truncate=True, max_length=270, cache_dir=cache_dir)

tokenizers = {"GPTJ": tokenizer, "GPT2": tokenizer_gpt2}

TOOL_START_TOKEN = "<TOOL>"
TOOL_END_TOKEN = "</TOOL>" 

tokenizer_gpt2.add_tokens([TOOL_START_TOKEN, TOOL_END_TOKEN])


def mask_tokenize_data(
        dataset_dir,
        output_dir,
        tool_name:str,
        model_names:list[str]=["GPTJ", "GPT2"],
):
    global tokenizers, TOOL_START_TOKEN, TOOL_END_TOKEN

    # We want to output: tokenized_start_text, tool_name, tokenized_end_text, token_type, start_method_A_train_mask, end_method_A_train_mask
    output_fields = ["tokenized_start_text", "tool_name", "tokenized_end_text", "start_token_type", "end_token_type", "start_method_A_train_mask", "end_method_A_train_mask"]

    # Create output directory
    for name in model_names:
        if not os.path.exists(f"{output_dir}/{name}"):
            os.makedirs(f"{output_dir}/{name}")

    file_list = [file for file in os.listdir(dataset_dir) if file.endswith('.csv')]
    for file in file_list:
        dataset = load_dataset(dataset_dir, split="train", data_files = file)
        data_iter = iter(dataset)

        # Iterate through the dataset and write to the output file
        data = next(data_iter, None)
        output = {}
        for name in model_names:
            output.update({key + name: [] for key in output_fields})
        while data is not None:
            
            text = data["API_call_response_text"]
            tokenized_text = {}
            for model_name in model_names:
                tokenized_text[model_name] = tokenizers[model_name](text, truncation=True, max_length=1000).input_ids

            # Find index where tokenized_text matches the tool start token:
            index_start_dict = {name: tokenized_text[name].index(tokenizers[name].encode(TOOL_START_TOKEN)[0]) for name in model_names}
            index_arrow_dict = {name: tokenized_text[name][index_start_dict[name]:].index(tokenizers[name].encode("→")[0]) + index_start_dict[name] + 1 for name in model_names}
            index_end_dict = {name: tokenized_text[name][index_arrow[name]:].index(tokenizers[name].encode(TOOL_END_TOKEN)[0]) + index_arrow[name] + 1 for name in model_names}

            len_start_dict = {name: index_start_dict[name] + 1 for name in model_names}

            len_toolname_dict = {name: len(tokenizers[name].encode(tool_name)) for name in model_names}

            # Find number of ocurrences of →
            occurrences = len(re.findall(r'(\)\→)', text))

            if occurrences != 1:
                print("More than one occurrence of →", flush=True)
                print(text, flush=True)
                raise Exception("More than one occurrence of →")

            # Create token type mask
            token_type_dict = {name: torch.zeros(len(tokenized_text[name])) for name in model_names}
            for name in model_names:
                mask = token_type_dict[name]
                len_start = len_start_dict[name]
                len_toolname = len_toolname_dict[name]
                index_arrow = index_arrow_dict[name]
                index_end = index_end_dict[name]

                mask[len_start] += 1                         # <TOOL>
                mask[len_start+1] += 1                       # Toolname
                mask[len_start+1+len_toolname] += 1        # (
                mask[len_start+1 + len_toolname + 1] += 1    # args
                mask[index_arrow - 1] += 1                   # )
                mask[index_arrow] += 1                       # →
                mask[index_arrow+1] += 1                     # response
                mask[index_end] += 1                         # </TOOL>
                mask[index_end+1] += 1                       # ...Data
                mask = mask.cumsum(dim=0)

                token_type_dict[name] = mask
                output["start_method_A_train_mask" + name].append(torch.isin(token_type_dict[name][:len_start+1], torch.tensor([0, 1])).view(-1).tolist())
                output["end_method_A_train_mask" + name].append(torch.isin(token_type_dict[name][len_start+1+len_toolname], torch.tensor([9])).view(-1).tolist())
            
                output["tokenized_start_text" + name].append(tokenized_text[name][:len_start+1])
                output["tool_name" + name].append(tool_name)
                output["tokenized_end_text" + name].append(tokenized_text[name][index_end+1:])
                output["start_token_type" + name].append(token_type_dict[name][:len_start+1].view(-1).tolist())
                output["end_token_type" + name].append(token_type_dict[name][len_start+1+len_toolname].view(-1).tolist())


            data = next(data_iter, None)
            


        for name in model_names:
            # Create output file
            with open(f"{output_dir}/{name}/{file}", 'w') as f:
                writer = DictWriter(f, fieldnames=output_fields)
                writer.writeheader()
                row = {key: output[key + name] for key in output_fields}
                writer.writerow(row)



if __name__ == "__main__":
    dataset_dir = "/vol/bitbucket/jg2619/data/augmented_standard/processed"
    output_dir = "/vol/bitbucket/jg2619/data/augmented_standard/test/train_ready/"
    mask_tokenize_data(dataset_dir, output_dir, "WikiSearch")



