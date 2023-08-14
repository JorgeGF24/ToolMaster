# In this file, we will conduct evaluation on diverse datasets for different models.
# We will test the Toolmaster model with several underlying models, including:
# 1. GPTJ with prompting enabling it to call external tools such as the calculator
# 2. LLAMA2 with prompting enabling it to call external tools such as the calculator
# 3. Our fine-tuned GPTJ model with short prompting


import csv
import datetime
import os
import sys
import json
import time
import re
import traceback

import torch
from tools import Calculator, calc_parse, WikiSearch, wiki_parse, Calendar, calend_parse

from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel, GPTJConfig, LlamaTokenizer, LlamaConfig, LlamaForCausalLM
from model_training.models.toolmaster import ToolMaster

from functools import partial

from copy import deepcopy

from beartype import beartype
from beartype.typing import List, Dict, Any, Union

Calculator = partial(Calculator, inference=True)


cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache"

TOOL_DOCUMENTATION = {"Calculator": """The calculator tool computes arithmetic expressions. You can call the API by writing "[Calculator(expression)]" where "expression" is the expression to be computed. Here are some examples of its usage:
Example 1: Last year we collected 237342 apples, double of what we collected this year: [Calculator(237342/2)→ 118671] 118671.
Example 2: The number in the next term is 18 + 12 x 3 = [Calculator(18+(12*3))→ 54] 54.
Example 3: A total of 252 matches were played, and 723 goals were scored (an average of [Calculator(723/252)→ 2.87] 2.87 per match). This is twenty goals more than the [Calculator(723-20)→703] 703 goals last year.
Example 4: I went to Paris in 1994 and stayed there until 2011, so in total, it was [Calculator(2011-1994)→ 17] 17 years.""",

"Calendar": """The calendar tool returns the current date. It can help you get information required to complete the text, such as the temporal context of a person, action or general information. You can call the API by writing "[Calendar( )]". Here are some examples of its usage:
Example 1: Today is the first [Calendar( )→ Today is Friday, 01/01/2019] Friday of the year.
Example 2: The president of the United States is [Calendar( )→ Today is Tuesday, 11/02/2007] George W. Bush.""",

"WikiSearch": """The WikiSearch tool retrives Wikipedia snipets. You can use it to look up encyclopedic information from the current context. You can do so by writing "[WikiSearch(term)]" where "term" is the search term you want to look up. Here are some examples of API calls:
Example 1: The colors on the flag of Ghana have the following meanings: red is for [WikiSearch("Ghana flag red meaning")] the blood of martyrs, green for forests, and gold for mineral wealth.
Example 2: But what are the risks during production of nanomaterials? [WikiSearch("nanomaterial production risks")] Some nanomaterials may give rise to various kinds of lung damage.
Example 3: Metformin is the first-line drug for [WikiSearch("Metformin first-line drug")] patients with type 2 diabetes and obesity."""}

TOOL_EXPLANATIONS = {tool_name:f"""{TOOL_DOCUMENTATION[tool_name]}

It can help you solve your current task. Now, complete the text below.

""" for tool_name in TOOL_DOCUMENTATION.keys()}


    # Tools given to the toolmaster must have:
    # 1. Name: str - Unique identifier for the tool
    # 2. Arg parser: Callable - A function that takes a string and returns a list of arguments
    # 3. Tool: Callable - A function that takes a list of argumets and returns a string
    # 4. Explanation prompt: Union[torch.Tensor, str] - A string that explains how to use the tool
    # 5. Short description: Optional[str] - A short description of the tool

TOOL_SPECS = {"Calculator":{
    "name": "Calculator",
    "arg_parser": lambda x: [calc_parse(x)],
    "tool": Calculator,
    "explanation_prompt": TOOL_EXPLANATIONS["Calculator"],
    "short_description": "adds, subtracts, multiplies and divides"
}, "Calendar":{
    "name": "Calendar", 
    "arg_parser": lambda x: [calend_parse(x)],
    "tool": Calendar,
    "explanation_prompt": TOOL_EXPLANATIONS["Calendar"],
    "short_description": "returns the current date"
}, "WikiSearch":{
    "name": "WikiSearch",
    "arg_parser": lambda x: [wiki_parse(x)],
    "tool": WikiSearch,
    "explanation_prompt": TOOL_EXPLANATIONS["WikiSearch"],
    "short_description": "searches Wikipedia"
}}

ANSWER_TOKEN_IDS = {"GPTJ": [33706, 41484, 23998, 3280],
                    "LLAMA": [673, 1234, 12011, 22550],}

POST_ANSWER_TOKEN_IDS = {"GPTJ": [628, 198],
                         "LLAMA": [13]}



################################## PROMPTS ####################################



FREE_GENERATION_PROMPT = {
    
# 1 shot with calculator explanation:
"CALC_EXPLAN_1SHOT": f"""You are a question answering model that can use external tools to answer questions. This is a demonstration of the Calculator tool:

{TOOL_DOCUMENTATION["Calculator"]}


You can use the following tools: [AVAILABLE TOOLS]. Now, answer the following questions. When you find the answer, write "Answer:" on a new line followed by your answer. For example, if the answer is 42, write "\\nAnswer: 42".

Question 1: Paris has 3 times the number of inhabitants as Madrid. Madrid has 1 million more inhabitants than Barcelona. Barcelona has 1.6 million inhabitants. How many inhabitants does Paris have?
Let's think step by step: Madrid has 1 million more inhabitants than Barcelona so it has [Calculator(1600000+1000000)→ 2600000] 2600000 inhabitants. Therefore, as Paris has three times Madrid's population, Paris has [Calculator(2600000*3)→ 7800000] 7800000 inhabitants.
Answer 1: 7800000

Question 2: [PROMPT]
Let's think step by step: """,

# 0 shot with calculator explanation:
"CALC_EXPLAN_0SHOT": f"""You are a question answering model that can use external tools to answer questions. This is a demonstration of the Calculator tool:

{TOOL_DOCUMENTATION["Calculator"]}


You can use the following tools: [AVAILABLE TOOLS]. Now, answer the following questions. When you find the answer, write "Answer:" on a new line followed by your answer. For example, if the answer is 42, write "\\nAnswer: 42".

Question: [PROMPT]
Let's think step by step: """,

# None
"None": None}


TRAINED_MODELS = {
    "GPTJ-no-add-sub-0": "/vol/bitbucket/jg2619/augmenting_llms/model_training/models/GPTJ_goody",

}


cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache"

MODEL_NAME = "GPTJ-bare"
BASE_MODEL_NAME = "GPTJ" # "GPTJ" or "LLAMA"

BENCHMARK_NAME = "gms8k-easy"

def load_GPTJ(path:str="EleutherAI/gpt-j-6B",
              new_tokens:List[str]=["[PAD]"],):
    # Load the GPTJ model we will use to construct the Toolmaster model
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir=cache_dir)

    tokenizer.add_tokens(new_tokens)
    tokenizer.pad_token=new_tokens[-1]

    config = GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B", padding_idx=tokenizer.pad_token_id)

    kwargs = {}
    if path == "EleutherAI/gpt-j-6B":
        kwargs = {"revision": "float16"}

    model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True, 
            config=config, 
            cache_dir=cache_dir, **kwargs).cuda()

    model.eval()
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

def load_LLAMA(path:str="meta-llama/Llama-2-7b-hf",
              new_tokens:List[str]=["[PAD]"],):
        
        kwargs = {cache_dir: cache_dir}
        if path == "meta-llama/Llama-2-7b-hf":
            kwargs = {"token":"***REMOVED***",}

        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf",
                                                   **kwargs)

        tokenizer.add_bos_token = False

        tokenizer.add_tokens(new_tokens)
        tokenizer.pad_token=new_tokens[-1]
        
        config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf", 
                                             padding_idx=tokenizer.pad_token_id,
                                             **kwargs)

        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",
                                                  torch_dtype=torch.float16,
                                                  low_cpu_mem_usage=True,
                                                  config=config,
                                                   **kwargs).cuda()

        model.eval()
        model.resize_token_embeddings(len(tokenizer))
        
        return model, tokenizer



def create_toolformer(
    config,
):
    # assert config has model, tokenizer and free_generation_prompt:
    assert "model" in config
    assert "tokenizer" in config
    assert "free_generation_prompt" in config


    # Provide defaults for max_new_tokens (40), 
    max_new_tokens = config.get("max_new_tokens", 40)

    output_dir = config.get("output_dir", ".")

    log_dir = output_dir + "/logs"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    tool_specs = []
    for tool in config["tools"]:
        tool_specs.append(TOOL_SPECS[tool])
    
    tool_token_ids = config.get("tool_token_ids", [])
    tool_tokens = config.get("tool_tokens", "[")
    if tool_token_ids == []:
        for key, value in config["tokenizer"].get_vocab().items():
            if any(token in key for token in tool_tokens):
                tool_token_ids.append(value)
    
    return ToolMaster(model = config["model"],
                      tokenizer = config["tokenizer"],
                      tool_specs = tool_specs,
                      tool_token_ids = tool_token_ids,
                      max_new_tokens = max_new_tokens,
                      free_generation_prompt = config["free_generation_prompt"],
                      log_dir=log_dir,
                      catch_answers=True,
                      answer_token_ids=ANSWER_TOKEN_IDS[BASE_MODEL_NAME],
                      post_answer_token_ids=POST_ANSWER_TOKEN_IDS[BASE_MODEL_NAME],
                      )

@beartype
def create_config(
            base_model_name: str = "GPTJ", # GPTJ or LLAMA
            model_path: str = "EleutherAI/gpt-j-6B",
            max_new_tokens: int = 40,
            free_gen_prompt_name: str = "CALC_EXPLAN_1SHOT",
            tools: list[str] = ["Calculator"],):
    global MODEL_NAME, FREE_GENERATION_PROMPT

    MODEL_NAME = "GPTJ-bare"

    if base_model_name == "GPTJ":
        model, tokenizer = load_GPTJ(model_path) 
    else:
        model, tokenizer = load_LLAMA(model_path)
        max_new_tokens = int(1.2*max_new_tokens)

    config = {
        "model": model,
        "tokenizer": tokenizer,
        "tools": tools,
        "free_generation_prompt": FREE_GENERATION_PROMPT[free_gen_prompt_name],
        "max_new_tokens": max_new_tokens,
    }
 
    return config



# Function that loads and returns the GMS8K dataset
def load_gms8k_easy():
    with open("ToolQA/data/questions/easy/gsm8k-easy.jsonl", "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    return data

def load_gms8k_hard():
    with open("ToolQA/data/questions/hard/gsm8k-hard.jsonl", "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    return data


def load_ASDiv():

    from bs4 import BeautifulSoup
    import re

    with open('/vol/bitbucket/jg2619/augmenting_llms/benchmarks/ASDiv/ASDiv.xml', 'r') as f:
        data = f.read()
    Bs_data = BeautifulSoup(data, "lxml")

    problem_ids = []

    with open('/vol/bitbucket/jg2619/augmenting_llms/benchmarks/ASDiv/fold0.txt', 'r') as f:
        for line in f.readlines():
            problem_ids.append(line.strip())

    data = []

    for id in problem_ids:
        problem = Bs_data.find("problem", id=id)
        question = str(problem.find(string=True, recursive=False)[3:-3] + " " + problem.question.text)
        answer = re.sub('[^0-9.]', '', problem.answer.text)

        data.append({"question":question, "answer":answer})

    return data

def load_triviaQA():
    with open('/vol/bitbucket/jg2619/augmenting_llms/benchmarks/TriviaQA/triviaqa-unfiltered/short-unfiltered-web-dev.json') as f:
        data = json.load(f)["Data"]
    return data


@beartype
@torch.no_grad()
def evaluate(
        model,
        questions,
        correct_answers,
        device="cuda",
        batch_size = 10,):
    
    model.eval()
    model.to(device)

    correct = 0
    total = len(data)

    total_time = 0

    model_outputs = []
    pending = questions

    i=0

    while len(pending) > 0:
        try:
            start_time = time.time()
            model_outputs += model(pending[:batch_size])
        except torch.cuda.OutOfMemoryError as e:             # type: ignore
            print("Out of memory error")
            print(f"Reducing batch size from {batch_size} to {batch_size-5}")
            batch_size -= 5
            continue
        
        for out in model_outputs:
            out["id"] += i
            
        i += batch_size
        total_time += time.time() - start_time
        pending = pending[batch_size:]

    responses = [output["response"] for output in model_outputs]

    # Save responses to a csv file with dict writer:
    with open(f"{BENCHMARK_NAME}-responses-{MODEL_NAME}.csv", "w") as f:
        # Write the header
        writer = csv.DictWriter(f, fieldnames=["question", "correct_answer", "response"])
        writer.writeheader()
        for out in model_outputs:
            row = {"question":out["user_prompt"], "correct_answer":correct_answers[out["id"]], "response":out["response"]}
            writer.writerow(row)

    for i in range(len(responses)):
        print(f"Question: {questions[i]}")
        print(f"Correct answer: {correct_answers[i]}")
        if i < len(responses):
            print(f"Response: {responses[i]}")
        if responses[i] is not None and str(correct_answers[i]) in responses[i]:
            correct += 1
            print("Correct!")
        print("Incorrect!")
    
    print(f"Accuracy: {correct/total}")
    print(f"Correct: {correct}")
    print(f"Total: {total}")

    model_answers = extract_model_answers(responses)

    return correct, total_time

# We want each response to contain:
# 1. The response from the model
# 2. A tool history:
#   a. The tool name
#   b. The tool arguments
#   c. The tool response
#   d. Status: Success or failure
# 3. Answer is correct

def prompt_model(model,
                 questions:List[str],
                 batch_size:int = 10,):
    
    total_time = 0

    model_outputs = []
    pending = questions

    while len(pending) > 0:
        try:
            start_time = time.time()
            model_outputs += model(pending[:batch_size])
        except torch.cuda.OutOfMemoryError as e:             # type: ignore
            print("Out of memory error")
            print(f"Reducing batch size from {batch_size} to {batch_size-5}")
            batch_size -= 5
            continue

        total_time += time.time() - start_time
        pending = pending[batch_size:]

    responses = [output["response"] for output in model_outputs]
    tool_histories = [output["tool_history"] for output in model_outputs]

    return responses, tool_histories, total_time


@beartype
def write_results(
    experiment_data:Union[Dict[List],List[Dict[str, Any]]],
    total_time=None,
):
    # count csv files in directory
    file_count = len([name for name in os.listdir('.') if name.endswith(".csv")])

    if isinstance(experiment_data, dict):
        # Transform dict to list of dicts:
        experiment_data = [dict(zip(experiment_data,t)) for t in zip(*experiment_data.values())]

    with open(f"{BENCHMARK_NAME}-responses-{MODEL_NAME}_{file_count}.csv", "w") as f:
        print(f"Writing responses to {BENCHMARK_NAME}-responses-{MODEL_NAME}_{file_count}.csv")
        
        # Standard writer:
        writer = csv.DictWriter(f, fieldnames=list(experiment_data[0].keys()))
        writer.writeheader()
        for row in experiment_data:
            writer.writerow(row)
            for key, value in row.items():
                print(f"{key}: {value}")
            print()

    with open(f"{BENCHMARK_NAME}-responses-{MODEL_NAME}_{file_count}.txt", "w") as f:
        # Write current date of experiment:
        f.write(f"Date: {datetime.datetime.now()}\n")
        if total_time is not None:
            # Write time taken for experiment:
            f.write(f"Time taken: {total_time}\n")


@beartype
def extract_model_answers(
        responses: list[str],
):
    # This function extracts the answers from the data. It assumes the model has correctly generated a sentence with "Answer(<response>)". We will extract the response from the sentence and return it.
    model_answers = []
    for i, response in enumerate(responses):
        # Check that the model has generated a sentence with "Answer(<response>)"
        if "Answer" not in response:
            print("The model has not generated a sentence with 'Answer'")
            print(f"Response {i}: {response}")
            model_answers.append("")
            continue
        
        # Extract the text after the first reference to Answer:
        response = response.split("Answer")[1]

        # Remove trailing full stop:
        response = response.strip(".")

        #response = response.split(")")[0]
        model_answers.append(response.strip())
    
    return model_answers

def first_number_in_answer(answers:list[str]):
    # We want to extract the first number in each answer:
    # Warn if there is an equal sign in the answer or more than one number
    first_numbers = []
    for answer in answers:
        try:
            numbers = re.findall(r'-?\d+(\.\d+)?', answer)
            if len(numbers) > 1:
                print(f"Warning: More than one number in answer: {answer}")
            first_number = numbers[0]
            # Equal sign in answer:
            if "=" in answer:
                print(f"Warning: Equal sign in answer: {answer}")
        except:
            first_number = ""
        first_numbers.append(first_number)

    return first_numbers

def stats(
        model_answers:List[str],
        correct_answers:List[str],
        tool_history:List[List[int]],
        answer_type:str=""
):
    
    print("Calculating stats...")
    TOOL_STATUS = ["", " correctly", " incorrectly"]
    REALMS = ["global", "exact", "includes"]

    examples_in_realm_count = {f"{realm}":0 for realm in REALMS}

    seen_tools = []

    # Tool history is a list of the history of tools used for each answer.
    # Each tool history is a list of tool ids.

    # We want the following stats:
    # 1. Exact match accuracy
    # 2. Include match accuracy
    # For each of the correct/incorrect groups, for both exact and includes:
    # We want stats on tool use:
    # 1. Number of tools used
    # 2. Tool type distribution
    # In total:
    # 1. Average tool use
    # 2. Average tool type distribution

    tool_stats = {
        "max number of tools used":0,
        "min number of tools used":1000,
    }
    tool_stats = {key+tool_status:value for key, value in tool_stats.items() for tool_status in TOOL_STATUS}

    default_stat = {
            "per tool stats":{},
    } | tool_stats

    stats = {realm:deepcopy(default_stat) for realm in REALMS}

    # Tool use stats:
    # Will add same stats per tool, in dictionary with key as tool id
    default_tool_use_count = {f"total {realm}" + tool_status:0 for realm in REALMS for tool_status in TOOL_STATUS}

    tool_use_count = deepcopy(default_tool_use_count)
    

    def compute_stats(tool_status, history_list, realm):
        nonlocal stats, tool_use_count
        print(f"Realm: {realm}, tool status: {tool_status}")
        print(f"STATS AT START: {stats}")

        tool_use_count[f"total {realm}" + tool_status] += len(history_list)
        stats[realm]["max number of tools used" + tool_status] = max(stats[realm]["max number of tools used" + tool_status], len(history_list))
        stats[realm]["min number of tools used" + tool_status] = min(stats[realm]["min number of tools used" + tool_status], len(history_list))
        for id in history_list:
            tool_use_count[id][f"total {realm}" + tool_status] += 1
            stats[realm]["per tool stats"][id]["max number of tools used" + tool_status] = max(stats[realm]["per tool stats"][id]["max number of tools used" + tool_status], history_list.count(id))
            stats[realm]["per tool stats"][id]["min number of tools used" + tool_status] = min(stats[realm]["per tool stats"][id]["min number of tools used" + tool_status], history_list.count(id))

        print(f"STATS AT END: {stats}")

    for i in range(len(model_answers)):
        print(f"Correct answer: {correct_answers[i]}")
        print(f"Response: {model_answers[i]}")

        use_list = [use["id"] for use in tool_history[i]]
        correct_use_list = [use["id"] for use in tool_history[i] if use["status"] == 0]
        incorrect_use_list = [use["id"] for use in tool_history[i] if use["status"] != 0]

        use_cases = [use_list, correct_use_list, incorrect_use_list]

        for tool_status, use_case in zip(TOOL_STATUS, use_cases):

            # Initialize per tool stats for new tools:
            for id in use_case:
                if id not in tool_use_count:
                    seen_tools.append(id)
                    tool_use_count[id] = deepcopy(default_tool_use_count)
                    for realm in REALMS:
                        stats[realm]["per tool stats"][id] = deepcopy(tool_stats)

            # Tool use stats:
            if tool_status == TOOL_STATUS[0]:
                examples_in_realm_count[REALMS[0]] += 1
            compute_stats(tool_status, use_case, REALMS[0])

            if str(correct_answers[i]) == model_answers[i]:
                realm = "exact"
                print("EXACT MAtch Correct!")
            elif str(correct_answers[i]) in model_answers[i] and len(str(correct_answers[i]).strip()) > 0:
                realm = "includes"
                print("Include Correct!")
            else:
                print("Incorrect answer")
                continue
                ## EXACT REALM

            # Tool use stats:
            if tool_status == TOOL_STATUS[0]:
                examples_in_realm_count[realm] += 1
            compute_stats(tool_status, use_case, realm)
                


    # Update averages in stats with counts:
    for realm in REALMS:
        realm_total = max(examples_in_realm_count[realm], 1)
        for tool_status in TOOL_STATUS:
            stats[realm]["average number of tools used" + tool_status] = tool_use_count[f"total {realm}" + tool_status]/realm_total
            stats[realm][f"total tools used" + tool_status] = tool_use_count[f"total {realm}" + tool_status]
            stats[realm][f"total examples" + tool_status] = examples_in_realm_count[realm]
            for id in seen_tools:
                stats[realm]["per tool stats"][id]["average number of tools used" + tool_status] = tool_use_count[id][f"total {realm}" + tool_status]/realm_total
                stats[realm]["per tool stats"][id][f"total tools used" + tool_status] = tool_use_count[id][f"total {realm}" + tool_status]
                stats[realm]["per tool stats"][id][f"total examples" + tool_status] = examples_in_realm_count[realm]

    # Accuracies:
    stats[REALMS[0]]["exact accuracy"] = examples_in_realm_count[REALMS[0]]/max(examples_in_realm_count["exact"], 1)
    stats[REALMS[0]]["includes accuracy"] = examples_in_realm_count[REALMS[0]]/max(examples_in_realm_count["includes"], 1)
    
    # Tree search through dictionary and print key branch and final leaf values:
    def print_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                print('  ' * indent + str(key))
                print_dict(value, indent+1)
            else:
                print('  ' * indent + str(key) + ": " + str(value))

    print_dict(stats)
    # Save stats to json file:
    # Count number of json files to give it an id:
    # Make stats dir in current directory if non existent:
    if not os.path.exists("./stats"):
        os.makedirs("./stats")
    file_id = len([f for f in os.listdir("./stats") if f.endswith(".json")])
    if answer_type != "":
        answer_type = "_" + answer_type
    with open(f"stats/stat_{file_id}"+answer_type+".json", "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Stats saved to stats/stat_{file_id}"+answer_type+".json")

    return stats


@beartype
def experiment(base_model: str,
               model_path:str=None,
               questions:List=None,
               tools: List[str]=["Calculator"],
               batch_size:int = 26,
               **kwargs):

    config_args = {
        "base_model_name":base_model,
        "path":model_path,
        "tools":tools,
    } | kwargs

    if model_path is None:
        del config_args["path"]
    config = create_config(**config_args)

    print(f"Config:")
    for key, value in config.items():
        print(f"{key}: {value}")
    print()
    toolmaster = create_toolformer(config)

    if questions is None:
        return toolmaster

    return prompt_model(toolmaster, questions, batch_size=batch_size)


if __name__ == "__main__":
    dataset_names = sys.argv[1].split(", ")
    experiment_names = sys.argv[2:]

    datasets = []
    # LOAD DATA:
    for dname in dataset_names:
        match dname.lower():
            case "gms8k-easy":
                data = {"data":load_gms8k_easy(),
                        "type":"math"}
            case "gms8k-hard":
                data = {"data":load_gms8k_hard(),
                        "type":"math"}
            case "asdiv":
                data = {"data":load_ASDiv(),
                        "type":"math"}
            case "triviaqa":
                data = {"data":load_triviaQA(),
                        "type":"wiki"}
            case "test":
                data = {"data":load_ASDiv()[:6],
                        "type":"math"}
            case _:
                raise ValueError(f"Dataset name {dname} not recognised")
            
        data["name"] = dname
        datasets.append(data)

    for name in experiment_names:

        for dataset in datasets:
            print(f"Running {name} experiment on dataset {dataset['name']}")

            ex_config = {
                "max_new_tokens": 100,
            }

            # Can be overriden by experiments:
            if dataset["type"] == "math":
                ex_config["tools"] = ["Calculator"]
            if dataset["type"] == "wiki":
                ex_config["tools"]["WikiSearch"]

            match name.upper():
                case "AY":
                    ex_config["base_model"] = "GPTJ"
                case "DX":
                    ex_config["base_model"] = "GPTJ"
                    ex_config["model_path"] = TRAINED_MODELS["GPTJ-no-add-sub-0"]
                case _:
                    raise ValueError(f"Experiment name {name} not recognised")
                
            #print(data)
            questions = [d["question"] for d in dataset["data"]]

            responses, tool_histories, total_time = experiment(
                questions=questions,
                **ex_config)
            
            model_answers = extract_model_answers(responses)

            correct_answers = [d["answer"] for d in dataset["data"]]
            ex_results = {"questions":questions,"correct_answers":correct_answers,"responses":responses, "model_answers":model_answers}

            if dataset["type"] == "math":
                ex_results["first_numbers"] = first_number_in_answer(model_answers)

            stats(model_answers=model_answers, correct_answers=correct_answers, tool_history=tool_histories, answer_type="extracted")
            stats(model_answers=responses, correct_answers=correct_answers, tool_history=tool_histories, answer_type="bare-responses")
            try:
                True
            except Exception as e:
                print(e)
                # Print traceback:
                traceback.print_exc()
                print("Stats failed to run")
            
            print(f"Finished experiment {name} for {dataset['name']}")
            print(ex_results)
            write_results(ex_results, total_time)


    raise SystemExit(42)
    performance_stats = {
        "Time to run model": 0,
        "Correct": 0,
    }
    
    performance_stats["Time to load model"] += time.time()

    # Evaluate the model on the GMS8K dataset
    correct, time_taken = evaluate(toolmaster, questions, answers)

    performance_stats["Time to run model"] = time_taken
    performance_stats["Correct"] = correct
