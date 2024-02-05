import json
import os

folder_path = '/home/***REMOVED***ccnet'
processed_folder_path = '/home/***REMOVED***ccnet_filenames'

# Create processed folder if it does not exist
os.makedirs(processed_folder_path, exist_ok=True)

for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        processed_data = []  # Initialize an empty list to store processed records
        with open(os.path.join(folder_path, filename), 'r') as f:
            for line in f:
                # Each line is a separate JSON object
                record = json.loads(line.strip())
                record['source_file'] = filename  # Add the filename as a field
                processed_data.append(record)  # Add the processed record to the list
                
        # Write the processed data to a new file, one JSON object per line
        with open(os.path.join(processed_folder_path, filename), 'w') as f:
            for record in processed_data:
                json_record = json.dumps(record)  # Convert the dict back to a JSON string
                f.write(json_record + '\n')  # Write the JSON string to file, adding a newline character
