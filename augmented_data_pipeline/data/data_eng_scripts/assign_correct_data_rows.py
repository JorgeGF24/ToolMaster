# The data produced by the toolformer pipeline has been incorrectly stored. The columns of the CSV are:
# url,text,API_calls_text,API_call_response_text,position,loss_improvement,arg_cohort,raw_arg,processed_arg,title,date_download,digest,length,nlines,source_domain,cc_segment,original_nlines,original_length,language,language_score,perplexity,bucket
#


# The columns of interes are "API_calls_text" and "API_call_response_text". We wish to match these columns with their correct original row. For this, we will read API_calls_text which has data in the form "data... [Calculator( .....)] ... data".
# We need to remove the text from [Calculator up to the next "]". The remaining text is our data and we need to match it with the "text" column of another file. From that matched row get the correct row information for our API_calls_text and API_call_response_text columns.

import re
import pandas as pd

dirty_file = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/new_horizon/augmented_standard/calculator_LLAMA/0.csv"
clean_file = dirty_file.replace(".csv", "_clean.csv")
correct_information_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/preprocessed/big_load_shuffled_nonewlines/calculator"
correct_files = ["9.csv"]

tool_name = "Calculator"

# Read the dirty file
dirty_df = pd.read_csv(dirty_file)

# Once we match the dirty file text (extracted from API_calls_text) with the correct file text (extracted from text), we substitute every column of the dirty file (except for API_calls_text and API_call_response_text) with the correct file's columns.

# Read the correct file
correct_df = pd.read_csv(correct_information_dir + "/" + correct_files[0])

i = 0
for row in dirty_df[1:]:
    print(f"Processing {i}")
    i += 1
    text = row["API_calls_text"]
    # Substitute  r'\[Calculator\([^]]*\)\]' for ' '
    text = re.sub(r' \[Calculator\([^]]*\)\]', '', text)

    # Match the text with the correct file's text
    correct_row = correct_df.loc[correct_df["text"] == text]
    print(f"Found text: {correct_row['text']}")

    # Substitute every column of the dirty file (except for API_calls_text and API_call_response_text) with the correct file's columns.
    for column in dirty_df:
        if column != "API_calls_text" and column != "API_call_response_text":
            row[column] = correct_row[column]

# Write the clean file
dirty_df.to_csv(clean_file, index=False)