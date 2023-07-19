# Go through files in a directory and substitute the substring '[' by '<TOOL>' if followed by toolname and ']' by '</TOOL>'

import os
import re
from csv import DictWriter, QUOTE_MINIMAL
from datasets import load_dataset


tools = ["Calendar", "Calculator", "WikiSearch"]

dataset_dir = "/vol/bitbucket/jg2619/data/augmented_standard/test/"

for tool in tools:
    file_list = [file for file in os.listdir(dataset_dir) if file.endswith('.csv')]

    # Create an output directory by adding "processed" to the input directory
    output_dir = os.path.join(dataset_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)

    for file in file_list:
        # Load dataset

        dataset = load_dataset(dataset_dir, split="train", data_files = file)
        columns = list(dataset.column_names)

        # Create a 

        # Create output file
        with open(f"{output_dir}/{file}", 'w') as f:
            writer = DictWriter(f, fieldnames=columns)
            writer.writeheader()

            i = 0

            for row in dataset:
                print(i)
                i += 1
                text = row["API_call_response_text"]
                text_no_resp = row["API_calls_text"]
                
                # regex expression that returns the text until where it matches the API start token followed by the tool name
                text = re.sub(rf"\[(?={tool[:4]})", '<TOOL>', text)
                text_no_resp = re.sub(rf"\[(?={tool[:4]})", '<TOOL>', text_no_resp)
                end_text = re.split(f"\)]", text_no_resp, maxsplit=1)
                if len(end_text) != 2:
                    print(end_text)
                    raise Exception("This is a stub")
                end_text = end_text[1]
                text = text[:-len(end_text)-1] + "</TOOL>" + end_text
                text_no_resp = text_no_resp[:-len(end_text)-1] + "</TOOL>" + end_text

                row["API_call_response_text"] = text
                row["API_calls_text"] = text_no_resp
                writer.writerow(row)
