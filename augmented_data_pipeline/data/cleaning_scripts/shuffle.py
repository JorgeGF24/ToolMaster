# This script shuffles the data in the csv file from a given directory and saves it to a new file in another directory.

import sys
import os
import csv
import random

csv.field_size_limit(sys.maxsize)

def shuffle(file_dir, output_dir):

    # create output directory
    os.makedirs(output_dir, exist_ok=True)

    file_list = [file for file in os.listdir(file_dir) if file.endswith('.csv')]
    for file in file_list:
        with open(os.path.join(file_dir, file), 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
            # print size of data
            print(len(data))
            header = data[0]
            data = data[1:]
            random.shuffle(data)
            data = [header] + data
            with open(os.path.join(output_dir, file), 'w') as f:
                writer = csv.writer(f)
                writer.writerows(data)





files_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/preprocessed/big_load_reverse/wikiSearch"
output_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/preprocessed/big_load_shuffled/wikiSearch"


shuffle(files_dir, output_dir)