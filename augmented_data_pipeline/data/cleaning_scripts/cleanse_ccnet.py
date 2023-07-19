# Read files 0.py, ..., 9.py in good_data/calendar and filter by the column named perplexity.
# The perplexity column is the 17th column, so we want to filter by the 17th column.
# Only keep rows with a perplexity < 80:

from csv import DictWriter
import json
import os
from datasets import load_dataset

dir = "/vol/bitbucket/jg2619/cc_net/data2/mined/2019-09_copy/"
new_dir = '/vol/bitbucket/jg2619/data/low_perplexity/'

max_perplexity = 100
max_lines = 20000

dataset = load_dataset(dir, split="train", streaming=True)
iter_data = iter(dataset)

header = ["url","raw_content","title","date_download","digest","length","nlines","source_domain","cc_segment","original_nlines","original_length","language","language_score","perplexity","bucket"]

counter = 0
file_counter = 0
# Create a new csv file
new_file = open(new_dir + str(file_counter) + ".csv", 'w', newline='')
# Write header to new csv file
csv_writer = DictWriter(new_file, fieldnames=header, escapechar='\\')
csv_writer.writeheader()

files = os.listdir(dir)
files.sort()



# Iterate through each row in the json file
for row in iter_data:

    #row = json.loads(json_dict)
    # The json file is a list of dictionaries, 
    # If the perplexity is less than max_perplexity
    if row['perplexity'] < max_perplexity:
        # Write the row to the new csv file
        csv_writer.writerow(row)
        counter += 1
        if counter % max_lines == 0:
            file_counter += 1
            new_file.close()
            new_file = open(new_dir + str(file_counter) + ".csv", 'w', newline='')
            csv_writer = DictWriter(new_file, fieldnames=header, escapechar='\\')
            csv_writer.writeheader()

new_file.close()