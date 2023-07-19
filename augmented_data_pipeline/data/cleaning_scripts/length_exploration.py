# This script tokenizes databases in a directory, saves the tokenized data and finds the length of the longest sequence.

import os
import csv
from transformers import AutoTokenizer


cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache"

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir=cache_dir)
print("Tokenizer loaded")

dir = "/vol/bitbucket/jg2619/data/preprocessed/low_perplexity/"
new_dir = "/vol/bitbucket/jg2619/data/preprocessed_tokenized/low_perplexity/"
tool = "calendar/"

max_len = 0
text_column = 1
# some metrics about the length of the tokenized text
average_len = 0
counter = 0

max_tokens = None

# Create new directory if it doesn't exist
if not os.path.exists(new_dir + tool):
    os.makedirs(new_dir + tool)

files = os.listdir(dir + tool)
print(f"Files: {files}")

def get_row():
    row = 0
    while row == 0:
        try:
            return next(reader, None)
        except:
            print(f"Error in file {file}")
            print(f"Line {counter_file}")
            row = 0
    return row

for file in files:
    if file.endswith(".csv"):
        counter_file = 0
        print(f"reading file {file}")
        # For each file in the directory
        with open(os.path.join(dir + tool, file)) as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader) # Skip header row

            with open(os.path.join(new_dir + tool, file), "w") as new_csv_file:
                writer = csv.writer(new_csv_file)
                header.append("tokenized_text")
                writer.writerow(header)

                row = get_row()
                while(row is not None):
                    # For each row in the file

                    # Tokenize the row
                    tokens = tokenizer.encode(row[text_column])

                    x = len(tokens)
                    average_len += x
                    counter += 1
                    counter_file += 1
                    if x > max_len:
                        max_len = x
                        print(f"New max length: {max_len}")
                        print(f"File: {file}")
                        print(tokens)
                        print()
                        max_tokens = tokens
                    
                    writer.writerow(row + [tokens])

                    row = get_row()

                
        print(f"Finished {file}")


print(f"Max length: {max_len}")
print(f"Average length: {average_len / counter}")
print(f"Max tokens: \n{max_tokens}")