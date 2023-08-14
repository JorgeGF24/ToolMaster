# This script goes through the files in a directory and escapes the new lines in the text column.

import os
import csv
import re

def escapechar_csv(file_dir):
    with open(file_dir, 'r') as f:
        reader = csv.DictReader(f)
        with open(file_dir[:-4] + "_escaped.csv", 'w') as f2:
            writer = csv.DictWriter(f2, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                # Replace '\n' with '\\n' IF the new line is not followed by "http"

                # This is a regex that matches a new line that is not followed by "http"
                row['text'] = re.sub(r'\n(?!(http))', '\\n', row['text'])
                writer.writerow(row)

if __name__ == "__main__":
    dir = "/vol/bitbucket/jg2619/data/preprocessed/big_load_reverse/wikiSearch/"
    
    for file in os.listdir(dir):  
        if file.endswith("copy.csv"):
            print("Processing " + file)
            escapechar_csv(dir + file)
            print("Done")