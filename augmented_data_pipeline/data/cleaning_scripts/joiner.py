# Script that joins csv files in a directory into bigger files of maximum size n.

import csv
import math
import os

def join(dir, new_dir, max_size=10000, names = None):
    # Iterate through each file in the directory
    lines = []
    files = []
    header = None
    for filename in os.listdir(dir):
        # check if it is a csv file
        if filename.endswith('.csv'):
            files.append(filename)
            # Count the number of lines in the file
            with open(dir + str(filename), 'r') as csv_file:
                csv_reader = csv.reader(csv_file)
                if not header:
                    header = next(csv_reader)
                n = sum(1 for line in csv_reader) - 1 # Subtract 1 for the header
                lines.append(n)
    # Calculate the number of files needed
    n_files = math.ceil(sum([n  for n in lines])/ max_size)
    if names is None:
        names = [str(i) + '.csv' for i in range(n_files)]
    else:
        names = names[:n_files]

    # Create the new files:
    for name in names:
        with open(new_dir + name, 'w') as new_file:
            csv_writer = csv.writer(new_file)
            csv_writer.writerow(header)

    # Iterate through each file in the directory
    written_lines = 0
    finished_files = 0
    current_file_read = 0
    maintain_current_file = False
    maintain_new_file = False
    while len(files) > 0:
        # Open the csv file
        if not maintain_current_file:
            current_file = files.pop(0)
        if not maintain_new_file:
            new_file_name = names.pop(0)
        maintain_current_file = False
        maintain_new_file = False
        with open(dir + str(current_file), 'r') as csv_file:
            # Read the csv file
            csv_reader = csv.reader(csv_file)
            header = next(csv_reader)
            # Create a new csv file

            with open(new_dir + new_file_name, 'a') as new_file:
                # Write to the new csv file
                csv_writer = csv.writer(new_file)
                # Skip header
                next(csv_reader)
                # Skip lines already read
                all(next(csv_reader) for _ in range(current_file_read))

                # Iterate through each row in the csv file
                for line in csv_reader:
                    # Write the row to the new csv file
                    csv_writer.writerow(line)
                    written_lines += 1
                    current_file_read += 1
                    if written_lines == max_size:
                        maintain_current_file = True
                        break
                if not maintain_current_file:
                    current_file_read = 0
                    maintain_new_file = True



if __name__ == '__main__':
    join('/vol/bitbucket/jg2619/data/augmented2/calendar2/', '/vol/bitbucket/jg2619/data/augmented2/calendar/', max_size=10000, names = [str(i) + '.csv' for i in range(10)])