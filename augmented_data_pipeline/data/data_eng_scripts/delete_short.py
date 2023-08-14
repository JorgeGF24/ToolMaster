# Go through the files in a directory and delete the ones that are too short:

import csv
import os
import sys
from datasets import load_dataset

dirs = ['/vol/bitbucket/jg2619/data/preprocessed/big_load_sorted/' + tool for tool in ['calendar/','calculator/', 'wikiSearch/']]
new_dirs = ['/vol/bitbucket/jg2619/data/preprocessed/big_load_sortshort/' + tool for tool in ['calendar/','calculator/', 'wikiSearch/']]


# load the dataset
for dir, new_dir in zip(dirs, new_dirs):
    # Iterate through each file in the directory
    for filename in os.listdir(dir):
        # Load the dataset
        csv_data = load_dataset(dir, data_files=[filename], split='train')
        # Read the csv file
        iter_data = iter(csv_data)
        # Check if new_dir exists and create it recursively if it doesn't
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        # Create a new csv file
        new_file = open(new_dir + filename, 'w')
        # Read header and identify perplexity column
        data = next(iter_data)
        header = list(data.keys())

        # Write to the new csv file
        csv_writer = csv.DictWriter(new_file, fieldnames=header)
        csv_writer.writeheader()

        while data is not None:
            if data['tokenized_length'] > 4:
                csv_writer.writerow(data)
            data = next(iter_data, None)
