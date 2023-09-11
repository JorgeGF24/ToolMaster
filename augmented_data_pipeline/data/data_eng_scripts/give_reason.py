# This script fills the selection_reason column of rows that have NaN in that column.

import pandas as pd
import numpy as np
import os
from csv import DictWriter

import re


def reason(
    sentence: str
) -> int:

    calc_pattern = re.compile("^(\d+(\s*[\+\-\*\/]\s*)?)+\d+$")
    phrases = ["=", "equals", "equal to", "total", "average of"]
    phrases_regex = '|'.join(map(re.escape, phrases))
    # Phrases followed by a number, decimal or with commas
    equals_pattern = rf'({phrases_regex})\s*((\d{1,3}(,\d{3})+)|(\d+))(\.\d+)?'

    nums = 0

    operators = bool(re.search(calc_pattern, sentence))
    equals = bool(re.search(equals_pattern, sentence))

    # 0 = random, 1 = operator, 2 = keywords, 3 = operation combination, 4 = operator and keywords

    if operators and equals:
        return 4
    elif equals:
        return 2
    elif operators:
        return 1

    #print("NO OPERATORS OR EQUALS")
    words = sentence.split(" ")
    numbers = []
    for word in words:
        if word.replace(".", "", 1).isnumeric():
            pass 
        elif word[:-1].replace(".", "", 1).isnumeric():
            word = word[:-1]  # remove commas, full stops, units, etc.
        elif word[1:].replace(".", "", 1).isnumeric():
            word = word[1:] # remove starting currency symbols
        else:
            continue
        try:
            num = float(word)
        except ValueError:
            print(f"error converting {word} to float")
            continue
        numbers.append(num)

    nums = len(numbers)
    
    if nums >= 3:
        #print("THREE NUMBERS MIN")
        if nums < 20:
            # Check if any of the numbers in words can be combined with +, -, *, / to produce a number in words
            for num1 in numbers:
                for num2 in numbers:
                    op_results = [num1 + num2, num1 - num2, num1 * num2]
                    if num2 != 0:
                        op_results.append(num1 / num2)
                    if any(result in numbers for result in op_results):
                        return 3
        
        return 0
    
    print("WTF")
    print(sentence)
    return -1
        


def give_reason(data_dir):

    # File list
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith("1.csv")]

    for file in files: 
        # Load data
        df = pd.read_csv(os.path.join(data_dir, file))

        # Clean file name:
        clean_dir = (data_dir[:-1] if data_dir.endswith("/") else data_dir) + "_clean/"

        # Writer object
        new_file = open(clean_dir + file, 'w')
        writer = DictWriter(new_file, fieldnames=df.columns)
        writer.writeheader()

        # Iterate through the rows
        for index, row in df.iterrows():
            if np.isnan(row['selection_reason']):
                code = reason(row['text'])
                row['selection_reason'] = code

            # Write row to file
            writer.writerow(row.to_dict())

        # close file
        new_file.close()
                


if __name__ == "__main__":
    give_reason("/vol/bitbucket/jg2619/data/preprocessed/big_load_reverse/calculator/")