# This script opens the 0.csv and reformats various columns.
# The columns to be reformatted are: 
# length, nlines, original_nlines, original_length, language_score,perplexity
# The data in these columns is in the format: 'tensor(DATA)', and we wish to convert it to just DATA.
# This can be done by doing x = x[7:-1], without the need for regex.
# The CSV has the columns:
# url,text,API_calls_text,API_call_response_text,loss_improvement,date,title,date_download,digest,length,nlines,source_domain,cc_segment,original_nlines,original_length,language,language_score,perplexity,bucket


import csv
import os

# Open the csv file
with open('0.csv', 'r') as csv_file:
    # Read the csv file
    csv_reader = csv.reader(csv_file)
    # Create a new csv file
    with open('0_clean.csv', 'w') as new_file:
        # Write to the new csv file
        csv_writer = csv.writer(new_file)
        # Iterate through each row in the csv file
        for line in csv_reader:
            # Iterate through each column in the row
            for i in range(len(line)):
                # If the column is one of the columns to be reformatted
                if i in [9, 10, 13, 14, 16, 17]:
                    # Remove the first 7 and last 1 characters
                    line[i] = line[i][7:-1]
            # Write the row to the new csv file
            csv_writer.writerow(line)
