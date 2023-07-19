from transformers import PreTrainedTokenizerBase
import dateutil.parser as dparser
import random
import re

# This file reads a list of json documents called 0000.json, 0001.json, etc.
# Each json document has the following fields:
# - url: the url of the webpage
# - raw_content: the text of the webpage
# - title: the title of the webpage
# - date_download: the date the webpage was downloaded "yyyy-mm-ddThh:mm:ssZ"
# - digest: the hash of the webpage
# - length: the length of the text in characters
# - nlines: the number of lines in the text
# - source_domain: the domain of the webpage
# - cc_segment: the Common Crawl segment id
# - original_nlines: the number of lines in the original text before CCNet processing
# - original_length: the length of the original text before CCNet processing
# - language: the language of the webpage "en" for all
# - language_score: the language score of the text
# - perplexity: the perplexity of the text
# - bucket: the bucket of the text (head, body, tail) (think all head?)


# Read data in batches of K
# Two options:
#  - tokenize data and preprocess it with heuristics in GPU
#  - preprocess data as strings
# Proccessed data is then stored in a csv file as strings or tokens.


class AvailableAPIs:
    """Keeps track of available APIs"""

    retrieval: bool = True
    calendar: bool = True
    calculator: bool = True
    llmchain: bool = True

    def check_any_available(self):
        return any([self.retrieval, self.calendar, self.calculator])


def check_apis_available(
    data: dict, tokenizer: PreTrainedTokenizerBase
) -> AvailableAPIs:
    """
    Returns available APIs with boolean flags

    :param data: from load_dataset, assumes ['text'] is available
    :param tokenizer: Tokenizer to tokenize data
    :return: AvailableAPIs
    """
    tokenized_data = tokenizer(data["text"])["input_ids"]
    available = AvailableAPIs()
    # In case we need a different version, found this here:
    # https://stackoverflow.com/questions/28198370/regex-for-validating-correct-input-for-calculator
    calc_pattern = re.compile("^(\d+[\+\-\*\/]{1})+\d+$")
    phrases = ["=", "equals", "equal to", "total of", "average of"]
    phrases_regex = '|'.join(map(re.escape, phrases))
    # Phrases followed by a number, decimal or with commas
    equals_pattern = rf'({phrases_regex})\s*((\d{1,3}(,\d{3})+)|(\d+))(\.\d+)?'

    if len(tokenized_data) < 4096:
        available.retrieval = False
    try:
        date = dparser.parse(data["url"], fuzzy=True)
    except (ValueError, OverflowError):
        available.calendar = False
    available.calculator = False
    # Activate calculator if more than 3 numbers or sentences like equal to, total of, etc
    tried_rand = False
    for i in range(len(tokenized_data) // 100):
        text = tokenizer.decode(tokenized_data[i * 100 : (i + 1) * 100])

        operators = bool(re.search(calc_pattern, text))
        equals = bool(re.search(equals_pattern, text))

        if not (operators and equals) and not tried_rand:
            tried_rand = True
            text = text.replace("\n", " ")
            text = text.split(" ")
            text = [item for item in text if item.replace(".", "", 1).isnumeric()]
            if len(text) >= 3:
                if random.randint(0, 99) == 0:
                    available.calculator = True
        else:
            available.calculator = True

    return available
