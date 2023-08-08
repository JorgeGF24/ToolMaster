# In this file, we will conduct evaluation on diverse datasets for different models.
# We will test the Toolmaster model with several underlying models, including:
# 1. GPTJ with prompting enabling it to call external tools such as the calculator
# 2. LLAMA2 with prompting enabling it to call external tools such as the calculator
# 3. Our fine-tuned GPTJ model with short prompting


import csv
import os
import sys
import json
import time

import torch
from tools import Calculator, calc_parse, WikiSearch, wiki_parse, Calendar, calendar_parse

from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel, GPTJConfig
from model_training.models.toolmaster import ToolMaster

from functools import partial

from beartype import beartype

Calculator = partial(Calculator, inference=True)


cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache"

TOOL_EXPLANATIONS = {"Calculator": """The calculator tool computes arithmetic expressions. You can call the API by writing "[Calculator(expression)]" where "expression" is the expression to be computed. Here are some examples of its usage:
Example 1: Last year we collected 237342 apples, double of what we collected this year: [Calculator(237342/2)→ 118671] 118671.
Example 2: The number in the next term is 18 + 12 x 3 = [Calculator(18+(12*3))→ 54] 54.
Example 3: A total of 252 matches were played, and 723 goals were scored (an average of [Calculator(723/252)→ 2.87] 2.87 per match). This is twenty goals more than the [Calculator(723-20)→703] 703 goals last year.
Example 4: I went to Paris in 1994 and stayed there until 2011, so in total, it was [Calculator(2011-1994)→ 17] 17 years.


It can help you solve your current task. Now, complete the text below.

""",

"Calendar": """The calendar tool returns the current date. It can help you get information required to complete the text, such as the temporal context of a person, action or general information. You can call the API by writing "[Calendar()]". Here are some examples of its usage:
Example 1: Today is the first [Calendar()→ Today is Friday, 01/01/2019] Friday of the year.
Example 2: The president of the United States is [Calendar()→ Today is Tuesday, 11/02/2007] George W. Bush.

It can help you solve your current task. Now, complete the text below.

""",

"WikiSearch": """The WikiSearch tool retrives Wikipedia snipets. You can use it to look up encyclopedic information from the current context. You can do so by writing "[WikiSearch(term)]" where "term" is the search term you want to look up. Here are some examples of API calls:
Example 1: The colors on the flag of Ghana have the following meanings: red is for [WikiSearch("Ghana flag red meaning")] the blood of martyrs, green for forests, and gold for mineral wealth.
Example 2: But what are the risks during production of nanomaterials? [WikiSearch("nanomaterial production risks")] Some nanomaterials may give rise to various kinds of lung damage.
Example 3: Metformin is the first-line drug for [WikiSearch("Metformin first-line drug")] patients with type 2 diabetes and obesity.

It can help you solve your current task. Now, complete the text below.

"""}


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
    "arg_parser": lambda x: [],
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


################################## PROMPTS ####################################



FREE_GENERATION_PROMPT = {
    
# 1 shot with calculator explanation:
"CALC_EXPLAN_1SHOT": f"""You are a question answering model that can use external tools to answer questions. This is a demonstration of the Calculator tool:

{TOOL_EXPLANATIONS["Calculator"]}


You can use the following tools: [AVAILABLE TOOLS]. Now, answer the following questions. When you find the answer, write "Answer:" on a new line followed by your answer. For example, if the answer is 42, write "\\nAnswer: 42".

Question 1: Paris has 3 times the number of inhabitants as Madrid. Madrid has 1 million more inhabitants than Barcelona. Barcelona has 1.6 million inhabitants. How many inhabitants does Paris have?
Let's think step by step: Madrid has 1 million more inhabitants than Barcelona so it has [Calculator(1600000+1000000)→ 2600000] 2600000 inhabitants. Therefore, as Paris has three times Madrid's population, Paris has [Calculator(2600000*3)→ 7800000] 7800000 inhabitants.
Answer 1: 7800000

Question 2: [QUESTION]
Let's think step by step: """,

# 0 shot with calculator explanation:
"CALC_EXPLAN_0SHOT": f"""You are a question answering model that can use external tools to answer questions. This is a demonstration of the Calculator tool:

{TOOL_EXPLANATIONS["Calculator"]}


You can use the following tools: [AVAILABLE TOOLS]. Now, answer the following questions. When you find the answer, write "Answer:" on a new line followed by your answer. For example, if the answer is 42, write "\\nAnswer: 42".

Question: [QUESTION]
Let's think step by step: """}







cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache"

MODEL_NAME = "GPTJ-bare"

BENCHMARK_NAME = "gms8k-easy"

def load_GPTJ():
    # Load the GPTJ model we will use to construct the Toolmaster model
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir=cache_dir)

    tokenizer.add_tokens(["[PAD]"])
    tokenizer.pad_token="[PAD]"

    config = GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B", padding_idx=tokenizer.pad_token_id)

    model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B",
            revision="float16",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True, config=config, cache_dir=cache_dir).cuda()

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
                      tool_specs = config["tool_specs"],
                      max_new_tokens = max_new_tokens,
                      free_generation_prompt = config["free_generation_prompt"],
                      log_dir=log_dir,
                      )

@beartype
def A_config(max_new_tokens: int = 40,
            free_gen_prompt_name: str = "CALC_EXPLAN_1SHOT",
            tools: list[str] = ["Calculator"],):
    global MODEL_NAME, FREE_GENERATION_PROMPT
    # Benchmark 1: GPTJ with prompting

    MODEL_NAME = "GPTJ-bare"

    model, tokenizer = load_GPTJ()

    config = {
        "model": model,
        "tokenizer": tokenizer,
        "tools": tools,
        "free_generation_prompt": FREE_GENERATION_PROMPT[free_gen_prompt_name],
        "max_new_tokens": max_new_tokens,
    }
 
    return config


@beartype
def extract_answers(
        responses: list[str],
):
    # This function extracts the answers from the data. It assumes the model has correctly generated a sentence with "Answer(<response>)". We will extract the response from the sentence and return it.
    answers = []
    for i, response in enumerate(responses):
        # Check that the model has generated a sentence with "Answer(<response>)"
        if "Answer" not in response:
            print("The model has not generated a sentence with 'Answer'")
            print(f"Response {i}: {response}")
            answers.append("")
            continue
        
        # Extract the text between the first reference to Answer and the next new line:
        response = response.split("Answer")[1].split("\n")[0]

        #response = response.split(")")[0]
        answers.append(response)
            
    return answers

# Function that loads and returns the GMS8K dataset
def load_gms8k_easy():
    with open("ToolQA/data/questions/easy/gsm8k-easy.jsonl", "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    return data

def load_gms8k_hard():
    with open("ToolQA/data/questions/hard/gsm8k-hard.jsonl", "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    return data


def load_asdiv():

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
        question = str(problem.find(text=True, recursive=False)[3:-3] + " " + problem.question.text)
        answer = re.sub('[^0-9.]', '', problem.answer.text)

        data.append({"question":question, "answer":answer})

    return data


def evaluate(
        model,
        questions,
        answers,
        device="cuda",
        batch_size = 10,):
    
    model.eval()
    model.to(device)

    correct = 0
    total = len(data)

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

    # Save responses to a csv file with dict writer:
    with open(f"{BENCHMARK_NAME}-responses-{MODEL_NAME}.csv", "w") as f:
        # Write the header
        writer = csv.DictWriter(f, fieldnames=["question", "answer", "response"])
        writer.writeheader()
        for out in model_outputs:
            row = {"question":out["original_prime"], "answer":answers[out["id"]], "response":out["response"]}
            writer.writerow(row)

    for i in range(len(responses)):
        print(f"Question: {questions[i]}")
        print(f"Answer: {answers[i]}")
        if i < len(responses):
            print(f"Response: {responses[i]}")
        if responses[i] is not None and str(answers[i]) in responses[i]:
            correct += 1
            print("Correct!")
        print("Incorrect!")
    
    print(f"Accuracy: {correct/total}")
    print(f"Correct: {correct}")
    print(f"Total: {total}")

    responses = extract_answers(responses)

    return correct, total_time

# We want each response to contain:
# 1. The response from the model
# 2. A tool history:
#   a. The tool name
#   b. The tool arguments
#   c. The tool response
#   d. Status: Success or failure
# 3. Answer is correct

def experiment_AX():

    toolmaster = A_config()





if __name__ == "__main__":
    model_name = sys.argv[1]
    dataset_name = sys.argv[2]
    match model_name:
        case "gptj":
            toolmaster = benchmark1()
        case "llama2":
            toolmaster = benchmark1()
        case _:
            raise ValueError(f"Model name {model_name} not recognised")
        
    match dataset_name:
        case "gms8k-easy":
            data = load_gms8k_easy()
        case "gms8k-hard":
            data = load_gms8k_hard()
        case "asdiv":
            data = load_asdiv()
        case _:
            raise ValueError(f"Dataset name {dataset_name} not recognised")


    questions = [d["question"] for d in data]
    answers = [d["answer"] for d in data]

    performance_stats = {
        "Time to load model": -time.time(),
        "Time to run model": 0,
        "Correct": 0,
    }
    
    performance_stats["Time to load model"] += time.time()

    # Evaluate the model on the GMS8K dataset
    correct, time_taken = evaluate(toolmaster, questions, answers)

    performance_stats["Time to run model"] = time_taken
    performance_stats["Correct"] = correct
