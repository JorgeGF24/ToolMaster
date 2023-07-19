# Go through the files in a directory and sort them by the tokenized length of the text column:

import csv
import os
import sys
from transformers import AutoTokenizer
from datasets import load_dataset
from csv import DictWriter
#'calculator/', 'calendar/'
dirs = ['/vol/bitbucket/jg2619/data/preprocessed/big_load/' + tool for tool in ['calculator/']]
new_dirs = ['/vol/bitbucket/jg2619/data/preprocessed/big_load_reverse/' + tool for tool in ['calculator/']]
cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache"

csv.field_size_limit(sys.maxsize)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", truncate=True, max_length=270, cache_dir=cache_dir)

for dir, new_dir in zip(dirs, new_dirs):
    # Iterate through each file in the directory
    for filename in os.listdir(dir):
        # check if it is a csv file
        # Check if file starts with 1, 6, 8 or 9:
        # start_check = filename in ['1.csv', '6.csv', '8.csv', '9.csv']
        if filename.endswith('1.csv'):
            # Open the csv file
            csv_data = load_dataset(dir, split='train', data_files=[filename])
            # Read the csv file
            iter_data = iter(csv_data)
            # Check if new_dir exists and create it recursively if it doesn't
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            # Create a new csv file
            new_file = open(new_dir + filename, 'w')
            # Read header and identify perplexity column
            data = next(iter_data)
            header = list(data.keys()) + ['tokenized_length']
            
            # Write to the new csv file
            csv_writer = DictWriter(new_file, fieldnames=header)
            csv_writer.writeheader()

            # Create a list of lines
            lines = []
            # Now we tokenize the text column and sort by length
            # Iterate through each row in the csv file
            while data is not None:
                # Tokenize the text column
                try:
                    len_tokenized_text = len(tokenizer.encode(data['text'], truncation=True, max_length=270))
                except:
                    print(data['text'])
                    data = next(iter_data, None)
                    continue
                data['tokenized_length'] = len_tokenized_text

                # Append line to list that we will then sort:
                lines.append(data)

                data = next(iter_data, None)

            # Sort the lines by the length of the tokenized text column
            lines.sort(key=lambda x: x['tokenized_length'], reverse=True)

            # Reverse the order of the list:
            #lines.reverse()

            # Write the sorted lines to the new csv file
            for line in lines:
                csv_writer.writerow(line)

            # Close the new csv file
            new_file.close()


