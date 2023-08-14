import os
import json
import csv

# Specify the input and output directories
input_directory = '/vol/bitbucket/jg2619/cc_net/data2/mined/2019-09 copy/'
output_directory = '/vol/bitbucket/jg2619/data/low_perplexity/'

# Iterate over the JSON files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.json'):
        json_path = os.path.join(input_directory, filename)
        csv_filename = filename[:-5] + '.csv'  # Change extension from .json to .csv
        csv_path = os.path.join(output_directory, csv_filename)

        # Open the JSON file and load the data
        with open(json_path) as json_file:
            data = json.load(json_file)

        # Open the CSV file for writing
        with open(csv_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)

            # Write the header row based on the keys of the first dictionary
            writer.writerow(data[0].keys())

            # Write each dictionary as a row in the CSV file
            for dictionary in data:
                writer.writerow(dictionary.values())

        print(f'Saved {csv_filename}')

print('Conversion complete!')