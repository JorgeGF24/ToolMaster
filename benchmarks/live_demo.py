from tools import Calculator, calc_parse, WikiSearch, wiki_parse, Calendar, calend_parse, GPT3Wiki
from transformers import AutoTokenizer, AutoModelForCausalLM

from functools import partial
import os

import torch


from model_training.models.toolmaster import ToolMaster

Calculator = partial(Calculator, inference=True)


cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache"

TOOL_DOCUMENTATION = {
    
"Calculator": {
    "tool_explanation":"""The calculator tool computes arithmetic expressions. You can call the API by writing "<TOOL>Calculator|expression→</TOOL>" where "expression" is the expression to be computed. Here are some examples of its usage:
Example 1: Last year we collected 237342 apples, double of what we collected this year: <TOOL>Calculator|237342/2→ 118671</TOOL> 118671.
Example 2: The number in the next term is 18 + 12 x 3 = <TOOL>Calculator|18+(12*3)→ 54</TOOL> 54.
Example 3: A total of 252 matches were played, and 723 goals were scored (an average of <TOOL>Calculator|723/252→ 2.87</TOOL> 2.87 per match). This is twenty goals more than the <TOOL>Calculator|723-20→703</TOOL> 703 goals last year.
Example 4: I went to Paris in 1994 and stayed there until 2011, so in total, it was <TOOL>Calculator|2011-1994→ 17</TOOL> 17 years.""",
    "few_shot_examples":[
"""Question: Paris has 3 times the number of inhabitants as Madrid, which has 1 million more inhabitants than Barcelona (1.6m inhabitants). How many inhabitants does Paris have?
Let's think step by step: Madrid has 1 million more inhabitants than Barcelona so it has <TOOL>Calculator|1.6+1→ 2.6</TOOL> 2.6m inhabitants. Therefore, as Paris has three times Madrid's population, Paris has <TOOL>Calculator|2.6*3→ 7.8</TOOL> 7.8m inhabitants.
Answer: 7.8 million""",
    ],
    "example_answer": "42"
},

"Calendar": {
    "tool_explanation":"""The calendar tool returns the current date. It can help you get information required to complete the text, such as the temporal context of a person, action or general information. You can call the API by writing "<TOOL>Calendar|</TOOL>". Here are some examples of its usage:
Example 1: Today is the first <TOOL>Calendar| → Today is Friday, 01/01/2019</TOOL> Friday of the year.
Example 2: The president of the United States is <TOOL>Calendar| → Today is Tuesday, 11/02/2007</TOOL> George W. Bush.""",
    "few_shot_examples":[
"""Question: How many days till the first of september?
Let's think step by step: Today is <TOOL>Calendar| → Today is Wednesday, 21/08/2023</TOOL> Wednesday. There are 31 days in agust, so there are <TOOL>Calculator|31-21→ 10</TOOL> 10 days left of August. The first of september is the day after the 31st, so there are 11 days left.
Answer: 11"""
        ],
    "example_answer": "2017"
},

"WikiSearch": {
    "tool_explanation":"""The WikiSearch tool retrives Wikipedia snipets. You can use it to look up encyclopedic information from the current context. You can do so by writing "<TOOL>WikiSearch|term→</TOOL>" where "term" is the search term you want to look up. Here are some examples of API calls:
Example 1: The colors on the flag of Ghana have the following meanings: red is for <TOOL>WikiSearch|Ghana flag red→ The red from Ghana's flag replesents the blood of martyrs</TOOL> the blood of martyrs, green for forests, and gold for mineral wealth.
Example 2: But what are the risks during production of nanomaterials? <TOOL>WikiSearch|nanomaterial production risk→ Evidence of lung deterioration</TOOL> Some nanomaterials may give rise to various kinds of lung damage.
Example 3: Metformin is the first-line drug for <TOOL>WikiSearch|Metformin drug use→ Metformin is used by diabetic patients</TOOL> patients with type 2 diabetes and obesity.
Example 4: The actress Jennifer Lawrence acted alongside Leonardo di Caprio in the movie <TOOL>WikiSearch|Jennifer Lawrence di Caprio movie→ They acted together in Don't Look up</TOOL> Don't Look Up.""",
    "few_shot_examples":[
"""Question: What year was the prime minister of England that served during WW2 born?
Let's think step by step: The prime minister of England that served during WW2 was <TOOL>WikiSearch|prime minister England WW2 born→ Winston Churchill</TOOL> Winston Churchill. He was born in <TOOL>WikiSearch|Winston Churchill born→ 1874</TOOL> 1874.
Answer: 1874"""
    ],
    "example_answer": "George Washington"
},

"GPT3Wiki": {
    "tool_explanation":"""The GPT3Wiki tool replies to user queries. It can reply to specific questions, or general information on a query. You can do so by writing "<TOOL>GPT3Wiki|query→</TOOL>" where "query" is the query you want to look up. Here are some examples of API calls:
Example 1: The colors on the flag of Ghana have the following meanings: red is for <TOOL>GPT3Wiki|meaning red Ghana flag→ The red from Ghana's flag replesents the blood of martyrs</TOOL> the blood of martyrs, green for forests, and gold for mineral wealth.
Example 2: But what are the risks during production of nanomaterials? <TOOL>GPT3Wiki|nanomaterial production risk→ Evidence of lung deterioration</TOOL> Some nanomaterials may give rise to various kinds of lung damage.
Example 3: Metformin is the first-line drug for <TOOL>GPT3Wiki|Metformin use→ Metformin is used by diabetic patients</TOOL> patients with type 2 diabetes and obesity.
Example 4: The actress Jennifer Lawrence acted alongside Leonardo di Caprio in the movie <TOOL>GPT3Wiki|What movie did Jennifer Lawrence and Leonardo di Caprio act together→ They acted together in Don't Look up</TOOL> Don't Look Up.""",
    "few_shot_examples":[
"""Question: What year was the prime minister of England that served during WW2 born?
Let's think step by step: The prime minister of England that served during WW2 was <TOOL>GPT3Wiki|prime minister England WW2→ Winston Churchill</TOOL> Winston Churchill. He was born in <TOOL>GPT3Wiki|Winston Churchill born→ 1874</TOOL> 1874.
Answer: 1874"""
    ],
    "example_answer": "George Washington"
}
}

ARG_PROMPT = """[TOOL_DOCUMENTATION]

It can help you solve your current task. Now, complete the text below.

[PROMPT]"""


TOOL_SPECS = {
    
"Calculator":{
    "name": "Calculator",
    "arg_parser": lambda x: [calc_parse(x)],
    "tool": Calculator,
    "explanation_prompt": ARG_PROMPT.replace("[TOOL_DOCUMENTATION]",  TOOL_DOCUMENTATION["Calculator"]["tool_explanation"]),
    "short_description": "can compute arithmetic expressions",
    "max_arg_length": 30,
},

"Calendar":{
    "name": "Calendar", 
    "arg_parser": lambda x: [calend_parse(x)],
    "tool": Calendar,
    "explanation_prompt": ARG_PROMPT.replace("[TOOL_DOCUMENTATION]",  TOOL_DOCUMENTATION["Calendar"]["tool_explanation"]),
    "short_description": "returns the current date",
    "max_arg_length": 1,

}, 
"WikiSearch":{
    "name": "WikiSearch",
    "arg_parser": lambda x: [wiki_parse(x)],
    "tool": WikiSearch,
    "explanation_prompt": ARG_PROMPT.replace("[TOOL_DOCUMENTATION]",  TOOL_DOCUMENTATION["WikiSearch"]["tool_explanation"]),
    "short_description": "returns explanations on a subject",
    "max_arg_length": 20,
}, 
"GPT3Wiki":{
    "name": "QuestionAnswerer",
    "arg_parser": lambda x: [wiki_parse(x)],
    "tool": GPT3Wiki,
    "explanation_prompt": ARG_PROMPT.replace("[TOOL_DOCUMENTATION]",  TOOL_DOCUMENTATION["GPT3Wiki"]["tool_explanation"]),
    "short_description": "answer any general knowledge questions",
    "max_arg_length": 20,
},
}


ANSWER_TOKEN_IDS = {"GPTJ": [33706, 41484, 23998, 3280, 24361],
                    "LLAMA": [673, 1234, 12011, 22550],}

POST_ANSWER_TOKEN_IDS = {"GPTJ": [628, 198],
                         "LLAMA": [13]}


PROMPT = """These are the available tools: 
[AVAILABLE TOOLS].

[USER_PROMPT]
"""

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir=cache_dir)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
        "/vol/bitbucket/jg2619/augmenting_llms/model_training/models/GPTJ_med_no_token2",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True, 
        #config=config, 
        cache_dir=cache_dir).cuda()

model.eval()

config = {
    "model": model,
    "tokenizer": tokenizer,
    "tools": ["Calculator", "Calendar", "WikiSearch", "GPT3Wiki"],
    "free_generation_prompt": PROMPT,
    "max_new_tokens": 70,
    "tool_top_k": 5,
    "pretty_tools": True,
    "tool_tokens": [" ["],
    "end_tool_token": "]"
}

def create_toolformer(
    config,
):
    output_dir = config.pop("output_dir", ".")
    log_dir = output_dir + "/logs"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    
    config["free_generation_prompt"] = config["free_generation_prompt"].replace("<TOOL>", "[")
    config["free_generation_prompt"] = config["free_generation_prompt"].replace("</TOOL>", "]")

    tool_specs = []
    for tool in config.pop("tools"):
        spec = TOOL_SPECS[tool]
        spec["explanation_prompt"] = spec["explanation_prompt"].replace("<TOOL>", "[")
        spec["explanation_prompt"] = spec["explanation_prompt"].replace("</TOOL>", "]")

        tool_specs.append(spec)
    
    tool_token_ids = config.pop("tool_token_ids", [])
    tool_tokens = config.pop("tool_tokens", "[")
    if tool_token_ids == []:
        for key, value in tokenizer.get_vocab().items():
            if any(token in key for token in tool_tokens):
                tool_token_ids.append(value)

    
    return ToolMaster(tool_specs = tool_specs,
                      tool_token_ids = tool_token_ids,
                      log_dir=log_dir,
                      catch_answers=True,
                      answer_token_ids=ANSWER_TOKEN_IDS["GPTJ"],
                      post_answer_token_ids=POST_ANSWER_TOKEN_IDS["GPTJ"],
                      **config
                      )

toolMaster = create_toolformer(config)

print(toolMaster(["Can you tell me about the civilization of Mesopotamia?"]))