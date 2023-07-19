# This python script reads the csv files 0.csv, 1.csv, ..., 9.csv in good_data/calendar and creates new csv files 0.csv, 1.csv, ..., 9.csv in good_data/calendar_short, with only two columns: text and date

import csv
import os

# Create the directory good_data/calendar_short if it does not exist
if not os.path.exists('good_data/calendar_short'):
    os.makedirs('good_data/calendar_short')

# Creates the files 0.csv, 1.csv, ..., 9.csv in good_data/calendar_short
for i in range(9):
    open('good_data/calendar_short/' + str(i) + '.csv', 'w').close()

# For each csv file in good_data/calendar
for i in range(9):
    with open('good_data/calendar/' + str(i) + '.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # Create a new csv file in good_data/calendar_short
        with open('good_data/calendar_short/' + str(i) + '.csv', 'w') as csvfile_short:
            writer = csv.writer(csvfile_short, delimiter=',')
            # For each row in the csv file
            for row in reader:
                # Write a new row with only the columns 1 and 2
                writer.writerow([row[1], row[2]])

