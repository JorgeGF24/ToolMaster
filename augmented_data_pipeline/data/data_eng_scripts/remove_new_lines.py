# This script removes new lines from the data.

import os
import re
from csv import DictWriter
from datasets import load_dataset


cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache"

def remove_new_lines(input_dir, output_dir):

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_list = [file for file in os.listdir(input_dir) if file.endswith('.csv') and file not in ["0.csv", "7.csv"]]

    for file in file_list:
        print(f"File is: {file}")
        dataset = load_dataset(input_dir, split="train", data_files = file, cache_dir=cache_dir)
        data_iter = iter(dataset)

        # Create output file
        with open(f"{output_dir}/{file}", 'w') as f:
            writer = DictWriter(f, fieldnames=dataset.column_names)
            writer.writeheader()

            # Iterate through the dataset and write to the output file
            data = next(data_iter, None)
            while data is not None:
                data["text"] = re.sub(r"\n", ". ", data["text"])
                writer.writerow(data)
                data = next(data_iter, None)


if __name__ == "__main__":
    # ORIGINALLY:
    # input_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/preprocessed/big_load_shuffled/"
    # NEW 1/08/2023:
    input_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/preprocessed/big_load/"
    output_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/preprocessed/big_load_nonewlines/"
    for tool in ["calendar","wikiSearch","calendar"]:
        remove_new_lines(input_dir + tool, output_dir + tool)