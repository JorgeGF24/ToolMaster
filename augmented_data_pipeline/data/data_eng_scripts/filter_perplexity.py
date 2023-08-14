# Read files 0.py, ..., 9.py in good_data/calendar and filter by the column named perplexity.
# The perplexity column is the 17th column, so we want to filter by the 17th column.
# Only keep rows with a perplexity < 80:

import csv
import os

def filter_by_perplexity(dir = 'good_data/calendar/', new_dir = 'perplexity_data/calendar/', perplexity = 80):
    # Iterate through each file in the directory
    for filename in os.listdir(dir):
        # check if it is a csv file
        if filename.endswith('.csv'):
            # Open the csv file
            with open(dir + filename, 'r') as csv_file:
                # Read the csv file
                csv_reader = csv.reader(csv_file)
                # Create a new csv file
                with open(new_dir + filename, 'w') as new_file:
                    # Write to the new csv file
                    csv_writer = csv.writer(new_file)
                    # Read header and identify perplexity column
                    header = next(csv_reader)
                    # Which column number is the perplexity column?
                    perp_row_num = header.index('perplexity')
                    # Write header to new csv file
                    csv_writer.writerow(header)

                    # Iterate through each row in the csv file
                    for line in csv_reader:
                        # If the perplexity is less than 50
                        if float(line[perp_row_num]) < perplexity:
                            # Write the row to the new csv file
                            csv_writer.writerow(line)

if __name__ == '__main__':
    filter_by_perplexity()