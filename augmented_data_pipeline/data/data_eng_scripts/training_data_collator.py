# This script readies the data for training. It takes the raw data from the dataset and converts it into a format that can be used by the model.

from csv import DictWriter
import json
import os
import torch
import re
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM

from torch.utils.data import DataLoader

cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache"


tokenizers = {"GPTJ": AutoTokenizer.from_pretrained("/vol/bitbucket/jg2619/models/tokenizer", truncate=True, max_length=270, cache_dir=cache_dir), 
              "GPT2": AutoTokenizer.from_pretrained("gpt2", truncate=True, max_length=270, cache_dir=cache_dir)}

TOOL_START_TOKEN = "<TOOL>"
TOOL_END_TOKEN = "</TOOL>" 

for name in tokenizers:
    #tokenizer_gpt2.add_tokens([TOOL_START_TOKEN, TOOL_END_TOKEN])
    tokenizers[name].add_tokens([TOOL_START_TOKEN, TOOL_END_TOKEN, "[PAD]"])





def mask_tokenize_data(
        data:dict,
        tool_name:str,
        model_names:list[str]=["GPTJ", "GPT2"],
        duplicity_count:dict={},
):
    global tokenizers, TOOL_START_TOKEN, TOOL_END_TOKEN

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
                                                             # 0 for data...
                mask[len_start] += 1                         # 1 for <TOOL>
                mask[len_start+1] += 1                       # 2 for Toolname
                mask[len_start+1+len_toolname] += 1          # 3 for (
                mask[len_start+1 + len_toolname + 1] += 1    # 4 for args
                mask[index_arrow - 1] += 1                   # 5 for )
                mask[index_arrow] += 1                       # 6 for →
                mask[index_arrow+1] += 1                     # 7 for response
                mask[index_end] += 1                         # 8 for </TOOL>
                mask[index_end+1] += 1                       # 9 for ...Data
                mask = mask.cumsum(dim=0)

                token_type_dict[name] = mask
                output["start_method_A_train_mask" + name].append(torch.isin(token_type_dict[name][:len_start+1], torch.tensor([0, 1])).view(-1).tolist())
                output["end_method_A_train_mask" + name].append(torch.isin(token_type_dict[name][len_start+1+len_toolname], torch.tensor([9])).view(-1).tolist())
            
                output["tokenized_start_text" + name].append(tokenized_text[name][:len_start+1])
                output["tool_name" + name].append(tool_name)
                output["tokenized_end_text" + name].append(tokenized_text[name][index_end+1:])
                output["start_token_type" + name].append(token_type_dict[name][:len_start+1].view(-1).tolist())
                output["end_token_type" + name].append(token_type_dict[name][len_start+1+len_toolname].view(-1).tolist())


            # This key is the decoded sentence from tokens of type 0 and 9
            # Extract this with masked select. Decode with tokenizers["GPTJ"].decode()
            duplicity_key = tokenizers["GPTJ"].decode(torch.masked_select(tokenized_text["GPTJ"], torch.isin(token_type_dict["GPTJ"], torch.tensor([0, 9]))))


            data = next(data_iter, None)
            



def prepare_training_data(dataset_dir:str,
                          output_dir:str, 
                          model_names:list[str]=["GPTJ", "GPT2"],
                          ):

    # We want to output: tokenized_start_text, tool_name, tokenized_end_text, token_type, start_method_A_train_mask, end_method_A_train_mask
    old_fields = ["url", "data_idx", "loss_improvement"]
    new_fields = ["tokenized_start_text", "tool_name", "tokenized_end_text", "start_token_type", "end_token_type", "start_method_A_train_mask", "end_method_A_train_mask"]

    # Create output directory
    for name in model_names:
        if not os.path.exists(f"{output_dir}/{name}"):
            os.makedirs(f"{output_dir}/{name}")

    # Create output file:
    for name in model_names:
        with open(f"{output_dir}/{name}/train.csv", 'w') as f:
            writer = DictWriter(f, fieldnames=old_fields+new_fields)
            writer.writeheader()

    # This dict keeps track of how many times a sentence comes up
    duplicity_count = {}


    file_list = [file for file in os.listdir(dataset_dir) if file.endswith('.csv')]
    for file in file_list:
        dataset = load_dataset(dataset_dir, split="train", data_files = file)

        dl = DataLoader(dataset, batch_size=1000, shuffle=False)
        data_iter = iter(dl)

        data = next(data_iter, None)

        while data is not None:
            train_data = mask_tokenize_data(data_iter, tool_name, model_names, duplicity_count)

            for name in model_names:
                # Create output file
                with open(f"{output_dir}/{name}/train.csv", 'a') as f:
                    writer = DictWriter(f, fieldnames=old_fields+new_fields)

                    for i, output_row in enumerate(train_data):
                        new_row = {key: output_row[key + name] for key in new_fields}
                        for key in old_fields:
                            new_row[key] = data[key][i]

                        writer.writerow(new_row)

            data = next(data_iter, None)


    # Save the duplicity count in a json file
    with open(f"{output_dir}/duplicity_count_{tool_name}.json", 'w') as f:
        json.dump(duplicity_count, f)



if __name__ == "__main__":
    dataset_dir = "/vol/bitbucket/jg2619/data/augmented_standard/processed"
    output_dir = "/vol/bitbucket/jg2619/data/augmented_standard/test/train_ready/"
    mask_tokenize_data(dataset_dir, output_dir, "WikiSearch")



