# In this file, we will conduct evaluation on diverse datasets for different models.
# We will test the Toolmaster model with several underlying models, including:
# 1. GPTJ with prompting enabling it to call external tools such as the calculator
# 2. LLAMA2 with prompting enabling it to call external tools such as the calculator
# 3. Our fine-tuned GPTJ model with short prompting


import csv
import datetime
import logging
import os
import random
import sys
import json
import time
import re
import traceback

import torch
from tools import Calculator, calc_parse, WikiSearch, wiki_parse, Calendar, calend_parse, GPT3Wiki

from model_training.models.toolmaster import ToolMaster

from functools import partial

from copy import deepcopy

from beartype import beartype
from beartype.typing import List, Dict, Any, Union, Callable


formatter = logging.Formatter('%(asctime)s:  %(message)s', datefmt='%m/%d/%Y %I:%M:%S  ')

eval_logs_dir = "./evaluation_logs"
if not os.path.exists(eval_logs_dir):
    os.makedirs(eval_logs_dir)

eval_logs_debug_dir = "./evaluation_logs_debug"
if not os.path.exists(eval_logs_debug_dir):
    os.makedirs(eval_logs_debug_dir)

STREAM_HANDLER = logging.StreamHandler(sys.stdout)
STREAM_HANDLER.setLevel(logging.WARN)
STREAM_HANDLER.setFormatter(formatter)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(STREAM_HANDLER)

TOOLMASTER_LOGGER = logging.getLogger("Toolmaster")

Calculator = partial(Calculator, inference=True)

cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache"

TOOL_DOCUMENTATION = {

"Calculator": {
    "tool_explanation":"""The calculator tool computes arithmetic expressions. You can call the API by writing "<TOOL>Calculator(expression→</TOOL>" where "expression" is the expression to be computed. Here are some examples of its usage:
Example 1: Last year we collected 237342 apples, double of what we collected this year: <TOOL>Calculator(237342/2)→ 118671</TOOL> 118671.
Example 2: The number in the next term is 18 + 12 x 3 = <TOOL>Calculator(18+(12*3))→ 54</TOOL> 54.
Example 3: A total of 252 matches were played, and 723 goals were scored (an average of <TOOL>Calculator(723/252)→ 2.87</TOOL> 2.87 per match). This is twenty goals more than the <TOOL>Calculator(723-20)→703</TOOL> 703 goals last year.
Example 4: I went to Paris in 1994 and stayed there until 2011, so in total, it was <TOOL>Calculator(2011-1994)→ 17</TOOL> 17 years.""",
    "task_shot_examples":[
        """Question: Mary had 23 apples. She gave 5 to John and 3 to Peter. How many apples does she have left?
Lets think step by step: she gave away a total of 5+3=8 apples, so she has 23-8=15 apples left.
Answer: 15"""],
    "few_shot_examples":[
"""Question: Paris has 3 times the number of inhabitants as Madrid. Madrid has 1 million more inhabitants than Barcelona. Barcelona has 1.6 million inhabitants. How many inhabitants does Paris have?
Let's think step by step: Madrid has 1 million more inhabitants than Barcelona so it has <TOOL>Calculator(1600000+1000000)→ 2600000</TOOL> 2600000 inhabitants. Therefore, as Paris has three times Madrid's population, Paris has <TOOL>Calculator(2600000*3)→ 7800000</TOOL> 7800000 inhabitants.
Answer: 7800000""",
    ],
    "example_answer": "42"
},

"Calendar": {
    "tool_explanation":"""The calendar tool returns the current date. It can help you get information required to complete the text, such as the temporal context of a person, action or general information. You can call the API by writing "<TOOL>Calendar(</TOOL>". Here are some examples of its usage:
Example 1: Today is the first <TOOL>Calendar( )→ Today is Friday, 01/01/2019</TOOL> Friday of the year.
Example 2: The president of the United States is <TOOL>Calendar( )→ Today is Tuesday, 11/02/2007</TOOL> George W. Bush.""",
    "task_shot_examples":[
        ""
    ],
    
    "few_shot_examples":[
"""Question: How many days till the first of september?
Let's think step by step: Today is <TOOL>Calendar( )→ Today is Wednesday, 21/08/2023</TOOL> Wednesday. There are 31 days in agust, so there are <TOOL>Calculator(31-21)→ 10</TOOL> 10 days left of August. The first of september is the day after the 31st, so there are 11 days left.
Answer: 11"""
        ],
    "example_answer": "2017"
},

"WikiSearch": {
    "tool_explanation":"""The WikiSearch tool retrives Wikipedia snipets. You can use it to look up encyclopedic information from the current context. You can do so by writing "<TOOL>WikiSearch(term)→</TOOL>" where "term" is the search term you want to look up. Here are some examples of API calls:
Example 1: The colors on the flag of Ghana have the following meanings: red is for <TOOL>WikiSearch(Ghana flag red meaning)→ The red from Ghana's flag replesents the blood of martyrs</TOOL> the blood of martyrs, green for forests, and gold for mineral wealth.
Example 2: But what are the risks during production of nanomaterials? <TOOL>WikiSearch(nanomaterial production risks)→ Evidence of lung deterioration</TOOL> Some nanomaterials may give rise to various kinds of lung damage.
Example 3: Metformin is the first-line drug for <TOOL>WikiSearch(Metformin first-line drug)→ Metformin is used by diabetic patients</TOOL> patients with type 2 diabetes and obesity.""",
    "few_shot_examples":[
"""Question: What year was the prime minister of England that served during WW2 born?
Let's think step by step: The prime minister of England that served during WW2 was <TOOL>WikiSearch(prime minister England WW2)→ Winston Churchill</TOOL> Winston Churchill. He was born in <TOOL>WikiSearch(Winston Churchill born)→ 1874</TOOL> 1874.
Answer: 1874"""
    ],
    "example_answer": "George Washington"
},
}


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





# Tools given to the toolmaster must have:
# 1. Name: str - Unique identifier for the tool
# 2. Arg parser: Callable - A function that takes a string and returns a list of arguments
# 3. Tool: Callable - A function that takes a list of argumets and returns a string
# 4. Explanation prompt: Union[torch.Tensor, str] - A string that explains how to use the tool
# 5. Short description: Optional[str] - A short description of the tool

MULTI_TOOL_USE_EXAMPLE = """Question: How many days till the first of september?
Let's think step by step: Today is <TOOL>Calendar| → Today is Wednesday, 21/08/2023</TOOL> Wednesday. There are 31 days in agust, so there are <TOOL>Calculator|31-21→ 10</TOOL> 10 days left of August. The first of september is the day after the 31st, so there are 11 days left.
Answer: 11"""

MULTI_TOOL_USE_EXAMPLE = """Question: How many days till the first of september?
Let's think step by step: Today is <TOOL>Calendar( )→ Today is Wednesday, 21/08/2023</TOOL> Wednesday. There are 31 days in agust, so there are <TOOL>Calculator(31-21)→ 10</TOOL> 10 days left of August. The first of september is the day after the 31st, so there are 11 days left.
Answer: 11"""


ANSWER_TOKEN_IDS = {"GPTJ": [33706, 41484, 23998, 3280, 24361],
                    "LLAMA": [673, 1234, 12011, 22550],}

POST_ANSWER_TOKEN_IDS = {"GPTJ": [628, 198],
                         "LLAMA": [13]}



################################## PROMPTS ####################################


FREE_GENERATION_PROMPT = {

####################    None    ################################# 0 0 0  None
"0 0 0": "[PROMPT]\n",

"0.5 wiki": """Answer the following questions that evaluate your general knowledge.

Question: [PROMPT]
""",
"0.5 math": """ Answer the following questions that asses your ability to calculate simple math problems.

Question: [PROMPT]
""",

################## basic_1-shot ################################# 0  0  n
"0 0 n": 
"""[FEW_SHOT_EXAMPLES]

Question: [PROMPT]
Let's think step by step: """,

################## TASK EXPLAN #################################   0 1   basic_1-shot
"0 1": """Answer the following questions. When you find the answer, write "Answer:" on a new line followed by your answer. For example, if the answer is [ANSWER_EXAMPLE], write "Answer: [ANSWER_EXAMPLE]".

[FEW_SHOT_EXAMPLES]

Question: [PROMPT]
Let's think step by step: """,
################## TASK EXPLAN #################################   0 1   basic_1-shot

"0 1 math": """Answer the following questions that asses your ability to calculate simple math problems.

Question: Mary had 23 flowers. She gave 5 to John and 3 to Peter. How many flowers does she have left?
She gave away a total of 5+3=8 flowers, so she has 23-8=15 flowers left. The answer is 15.

Question: [PROMPT]
""",

"0 1 wiki":"""Answer the following questions that evaluate your general knowledge.

Question: What country is the Burj Khalifa in?
Answer: It is in Dubai, so in the answer is United Arab Emirates.

Question: [PROMPT]
""",

"0 1+ math": """Answer the following questions that asses your ability to calculate simple math problems.

Question: Paris has 3 times the number of inhabitants as Madrid, which has 1 million more inhabitants than Barcelona (1.6m inhabitants). How many inhabitants does Paris have?
Answer: Madrid has 1 million more inhabitants than Barcelona so it has 1.6m + 1m = 2.6m inhabitants. Therefore, as Paris has three times Madrid's population, Paris has 2.6m*3 = 7.8m inhabitants. The answer is 7.8 million.

Question: How many days till the first of september?
Answer: Today is 21/08. There are 31 days in agust, so there are 31-21 = 10 days left of August. The first of september is the day after the 31st, so there are 11 days left. The answer is 11.

Question: [PROMPT]
Answer:""",

"0 1+ wiki": """Answer the following questions that evaluate your general knowledge.

Question: How many days till the first of september?
Answer: Today is 21/08. There are 31 days in agust, so there are 10 days left of August. The first of september is the day after the 31st, so there are 11 days left. The answer is 11.

Question: What year was the prime minister of England that served during WW2 born?
Answer: The prime minister of England that served during WW2 was Winston Churchill. He was born in 1874. The answer is 1874.

Question: [PROMPT]
Answer:""",

"0 1b math": """Answer the following questions that asses your ability to calculate simple math problems.

Question: Mary had 23 flowers. She gave 5 to John and 3 to Peter. How many flowers does she have left?
Answer: 15.

Question: [PROMPT]
""",

"0 1b wiki":"""Answer the following questions that evaluate your general knowledge.

Question: What country is the Burj Khalifa in?
Answer: United Arab Emirates.

Question: [PROMPT]
""",

    
############### TOOL EXPLAN, TASK EXPLAN #######################   1 1
"1 1": f"""You are a question answering model that can use external tools to answer questions. This is a demonstration of the [TOOL_NAME] tool:

[TOOL_DOCUMENTATION]


You can use the following tools: [AVAILABLE TOOLS].. Now, answer the following questions. When you find the answer, write "Answer:" on a new line followed by your answer. For example, if the answer is [ANSWER_EXAMPLE], write "Answer: [ANSWER_EXAMPLE]".

[FEW_SHOT_EXAMPLES]

Question: [PROMPT]
Let's think step by step: """,

"1 1 math": f"""You are a question answering model that can use external tools to answer questions. This is a demonstration of the [TOOL_NAME] tool:

[TOOL_DOCUMENTATION]


 Answer the following questions that asses your ability to calculate simple math problems.

Question: Mary had 23 flowers. She gave 5 to John and 3 to Peter. How many flowers does she have left?
She gave away a total of 5+3=8 flowers, so she has 23-8=15 flowers left. The answer is 15.

Question: Initially, I had $106 in my account. I used half of it to buy a lamp. How much do I have left?
Half of 106 is 53, so the answer is 53.

Question: [PROMPT]
""",

"1 1 wiki": f"""You are a question answering model that can use external tools to answer questions. This is a demonstration of the [TOOL_NAME] tool:

[TOOL_DOCUMENTATION]


Answer the following questions that evaluate your general knowledge.

Question: How many eyes do Koalas have?
Koalas are mammals, and mammals have 2 eyes. The answer is 2.

Question: [PROMPT]
""",

"1 1b math": f"""You are a question answering model that can use external tools to answer questions. This is a demonstration of the [TOOL_NAME] tool:

[TOOL_DOCUMENTATION]


 Answer the following questions that asses your ability to calculate simple math problems.

Question: Mary had 23 flowers. She gave 5 to John and 3 to Peter. How many flowers does she have left?
Answer: 15.

Question: [PROMPT]
""",

"1 1b wiki": f"""You are a question answering model that can use external tools to answer questions. This is a demonstration of the [TOOL_NAME] tool:

[TOOL_DOCUMENTATION]


Answer the following questions that evaluate your general knowledge.

Question: What country is the Burj Khalifa in?
Answer: United Arab Emirates.

Question: [PROMPT]
""",

"1.5 math": f"""You are a question answering model that can use external tools to answer questions. This is a demonstration of the [TOOL_NAME] tool:

[TOOL_DOCUMENTATION]


 Answer the following questions that asses your ability to calculate simple math problems.

Question: [PROMPT]
Answer:""",

"1.5 wiki": f"""You are a question answering model that can use external tools to answer questions. This is a demonstration of the [TOOL_NAME] tool:

[TOOL_DOCUMENTATION]


Answer the following questions that evaluate your general knowledge.

Question: [PROMPT]
Answer:""",

"1 2 math": f"""You are a question answering model that can use external tools to answer questions. This is a demonstration of the [TOOL_NAME] tool:
[TOOL_DOCUMENTATION]

 Answer the following questions that asses your ability to calculate simple math problems.

Question: Paris has 3 times the number of inhabitants as Madrid, which has 1 million more inhabitants than Barcelona (1.6m inhabitants). How many inhabitants does Paris have?
Answer: Madrid has 1 million more inhabitants than Barcelona so it has [Calculator|1.6+1→ 2.6] 2.6m inhabitants. Therefore, as Paris has three times Madrid's population, Paris has [Calculator|2.6*3→ 7.8] 7.8m inhabitants. The answer is 7.8 million.

Question: How many days till the first of september?
Answer: Today is [Calendar| → Today is Wednesday, 21/08/2023] Wednesday. There are 31 days in agust, so there are [Calculator|31-21→ 10] 10 days left of August. The first of september is the day after the 31st, so there are 11 days left. The answer is 11.

Question: [PROMPT]
Answer:""",

"1 2 wiki": f"""You are a question answering model that can use external tools to answer questions. This is a demonstration of the [TOOL_NAME] tool:
[TOOL_DOCUMENTATION]

Answer the following questions that evaluate your general knowledge.

Question: How many days till the first of september?
Answer: Today is [Calendar| → Today is Wednesday, 21/08/2023] Wednesday. There are 31 days in agust, so there are [Calculator|31-21→ 10] 10 days left of August. The first of september is the day after the 31st, so there are 11 days left. The answer is 11.

Question: What year was the prime minister of England that served during WW2 born?
Answer: The prime minister of England that served during WW2 was [WikiSearch|prime minister England WW2→ Winston Churchill] Winston Churchill. He was born in [WikiSearch|Winston Churchill born→ 1874] 1874. The answer is 1874.

Question: [PROMPT]
Answer:""",

############ TOOL EXPLAN, ARG PROMPT #######################  1 2 TOOL_EXPLAN_SHORT

"ARG_PROMPT": """[TOOL_DOCUMENTATION]

It can help you solve your current task. Now, complete the text below.

[PROMPT]""",


################# AVAILABLE TOOLS BARE ##############################  2  0  0  toolmaster-finetuned-bare
"2 0 0":"""These are the available tools: 
[AVAILABLE TOOLS].

Question: [PROMPT]
Answer:""",

################# AVAILABLE TOOLS BARE ##############################  2  0  0  toolmaster-finetuned-bare
"2 00 0":"""These are the available tools: 
[AVAILABLE TOOLS].

[PROMPT] 
Answer:""",

################# AVAILABLE TOOLS BARE ##############################  2  0  n   toolmaster-finetuned-few-shot
"2 0 n":"""These are the available tools: 
[AVAILABLE TOOLS].

[FEW_SHOT_EXAMPLES]

Question: [PROMPT]
Let's think step by step: """,


#################### AVAILABLE TOOLS, TASK  ########################  2  1   toolmaster-finetuned-task
"2 1":"""These are the available tools: 
[AVAILABLE TOOLS].

Now, answer the following questions. When you find the answer, write \"Answer:\" followed by your answer. For example, if the answer is [ANSWER_EXAMPLE], write \"Answer: [ANSWER_EXAMPLE]\".

[FEW_SHOT_EXAMPLES]

Question: [PROMPT]
Let's think step by step: """,

"2 0 2 math":"""These are the available tools: 
[AVAILABLE TOOLS].

Question: John has 78 dishes. Mary gives him 21 more. Sadly, one third of them break. How many dishes does John have left?
Answer: John has 78+21 = <TOOL>Calculator|78+21→ 99.0</TOOL> dishes. One third of them break, so two thirds are left: <TOOL>Calculator|99*2/3→ 66.0</TOOL> 66. The answer is 66.

Question: Paris has 3 times the number of inhabitants as Madrid, which has 1 million more inhabitants than Barcelona (1.6m inhabitants). How many inhabitants does Paris have?
Answer: Madrid has 1 million more inhabitants than Barcelona so it has <TOOL>Calculator|1.6+1→ 2.6</TOOL> 2.6m inhabitants. Therefore, as Paris has three times Madrid's population, Paris has <TOOL>Calculator|2.6*3→ 7.8</TOOL> 7.8m inhabitants. The answer is 7.8 million.

Question: How many days till the first of september?
Answer: Today is <TOOL>Calendar| → Today is Wednesday, 21/08/2023</TOOL> Wednesday. There are 31 days in agust, so there are <TOOL>Calculator|31-21→ 10</TOOL> 10 days left of August. The first of september is the day after the 31st, so there are 11 days left. The answer is 11.

Question: [PROMPT]
Answer:""",

"2 0 2 wiki":"""These are the available tools: 
[AVAILABLE TOOLS].

Question: How many days till the first of september?
Answer: Today is <TOOL>Calendar| → 21/08/2023</TOOL> 21/08. There are 31 days in agust, so there are <TOOL>Calculator|31-21→ 10</TOOL> 10 days left. The first of september is the day after the 31st, so there are 11 days left. The answer is 11.

Question: What year was the prime minister of England that served during WW2 born?
Answer: The prime minister of England that served during WW2 was <TOOL>WikiSearch|prime minister England WW2→ Winston Churchill</TOOL> Winston Churchill. He was born in <TOOL>WikiSearch|Winston Churchill born→ 1874</TOOL> 1874. The answer is 1874.

Question: Who sang the song "I wish you were here"?
Answer: The song "I wish you were here" was sung by <TOOL>WikiSearch|I wish you were here song→ Pink Floyd</TOOL> Pink Floyd. The answer is Pink Floyd.

Question: [PROMPT]
Answer:""",

"2 1 math":"""These are the available tools: 
[AVAILABLE TOOLS].

Answer the following questions that asses your ability to calculate simple math problems.

Question: [PROMPT]
Answer:""",

"2 1 wiki":"""These are the available tools: 
[AVAILABLE TOOLS].

Answer the following questions that evaluate your general knowledge.

Question: [PROMPT]
Answer:""",


#################### AVAILABLE TOOLS, TASK  ########################  2  1   toolmaster-finetuned-task
#Question: Mary had 23 apples. She gave 5 to John and 3 to Peter. How many apples does she have left?\nShe gave away a total of 5+3=8 apples, so she has 23-8=15 apples left. The answer is 15.
#Question: Who is the guitarist of the band Queen?\nThe guitarist is Brian May. The answer is Brian May.
"2 3 math":"""These are the available tools: 
[AVAILABLE TOOLS].

 Answer the following questions that asses your ability to calculate simple math problems.

Question: Mary had 23 flowers. She gave 5 to John and 3 to Peter. How many flowers does she have left?
She gave away a total of 5+3=8 flowers, so she has 23-8=15 flowers left. The answer is 15.

Question: Elsa and I have a total of $105. I have 4 times more dollars than her. How many dollars does Elsa have?
Let Elsa have $x, so I have $4*x. We know that x+4x=105, so 5x=105 and x=105/5=21. The answer is 21.

Question: John is putting apples in boxes of 6. He has 41 apples. How many boxes will he need?
He will need 41/6 = 6.83 boxes, so he will fill 6 boxes and need one more for the left-over apples. The answer is 7.

Question: [PROMPT]
""",

"2 3ddd wiki":"""These are the available tools: 
[AVAILABLE TOOLS].

Answer the following questions that evaluate your general knowledge.

Question: What is the loudest animal on Earth?
The blue whale. 
Answer: The loudest animal on Earth is the blue whale.

Question: What year was the prime minister of England that served during WW2 born?
During WW2, the English prime minister was Winston Churchill. He was born in 1874
Answer: The prime minister of England that served during WW2 was born in 1874.

Question: [PROMPT]
""",

"2 3 wiki":"""These are the available tools: 
[AVAILABLE TOOLS].

Answer the following questions that evaluate your general knowledge.

Question: What is the loudest animal on Earth?
Answer: The loudest animal on Earth is the blue whale.

Question: What year was the prime minister of England that served during WW2 born?
Answer: Wiston Churchill served England during WW2 and he was born in 1874.

Question: [PROMPT]
Answer:""",

"2 36 math":"""These are the available tools: 
[AVAILABLE TOOLS].

 Answer the following questions that asses your ability to calculate simple math problems.

Question: Mary had 23 flowers. She gave 5 to John and 3 to Peter. How many flowers does she have left?
She gave away a total of 5+3=8 flowers, so she has 23-8=15 flowers left. The answer is 15.

Question: Initially, I had $106 in my account. I used half of it to buy a lamp. How much do I have left?
Half of 106 is 53, so the answer is 53.

Question: Example 1: Last year we collected 237342 apples, double of what we collected this year. How many apples did we collect this year?
This year we collected 237342/2 = 118671, so the answer is 118671.

Question: What is 18 + 12 x 3?
The answer is 18+(12*3) = 54.

Question: A total of 252 matches were played, and 723 goals were scored. What is the average goals per match?
The average is the number of goals divided by the number of matches, so 2.87. The answer is 2.87.

Question: I went to Paris in 1994 and stayed there until 2011. How many years did I live in Paris?
In total, it was 2011-1994= 17 years. The answer is 17.

Question: [PROMPT]
""",

"2 36 wiki":"""These are the available tools: 
[AVAILABLE TOOLS].

Answer the following questions that evaluate your general knowledge.

Question: How many eyes do Koalas have?
Koalas are mammals, and mammals have 2 eyes. The answer is 2.

Question: What does the green color on the flag of Ghana meaning?
The green represents forests, so the answer is forests.

Question: What are the risks during production of nanomaterials? 
Some nanomaterials may give rise to various kinds of lung damage, so the answer is lung damage.

Question: What patiens use Metformin?
The answer is patients with type 2 diabetes and obesity.

Question: [PROMPT]
""",

"gms p":"""These are the available tools: 
[AVAILABLE TOOLS].

Answer the following difficult math questions.

Question: My brother is half my age. In 10 years, he will be 3/4 of my age. How old am I?
Answer: Let my age be x and my brother's y. Then, x*1/2 = y and (x+10)*3/4 = y+10. Solving this system we get [WolframAlphaCalculator| x*1/2 = y; (x+10)*3/4 = y+10 → x=10, y=5] that my age is 10 and my brother is 5. The answer is 10.

Question: In a basket, there are 24 good oranges and rest are bad oranges. The ratio of good oranges to the bad oranges is 3:1.
Answer: For each 3 good oranges there is 1 bad orange, so there are 24/3 = [WolframAlphaCalculator|24/3 → 8] 8 bad oranges. The answer is 8.

Question: Charlie has 6 apples at the start of the day. He gives some to Bob, who had none, and Bob gives half of those to Alice. At the end of the day, Alice has 2 apples less than Charlie. How many apples did Bob get?
Answer: We note that Alice and Bob have the same number of apples at the end of the day, as Bob gave half to Alice. Let x be the apples Alice and Bob have at the end of the day. We have a total of 6 apples, and Alice has x, Bob has x, and Charlie has x+2. Therefore, 6 = x + x + x+2, so x = [WolframAlphaCalculator|6 = x + x + x+2 → 1] 1. If Bob now has 1 apple after giving half of them, he got 1*2 = 2 apples from Charlie. The answer is 2.

Question: [PROMPT]
Answer:""",


#################### AVAILABLE TOOLS, TASK  ########################  2  1   toolmaster-finetuned-task
"2 3b":"""These are the available tools: 
[AVAILABLE TOOLS].

The following questions are meant to asses your ability to calculate simple math problems. When you find an answer, write \"Answer:\" followed by your answer. For example, if the answer is [ANSWER_EXAMPLE], write \"Answer: [ANSWER_EXAMPLE]\".

[FEW_SHOT_EXAMPLES]

Question: [PROMPT]
Let's think step by step: """,

}

def prepare_prompt(prompt_name, tool_name=None, few_shot_n=0):
    global FREE_GENERATION_PROMPT, TOOL_DOCUMENTATION, TOOL_SPECS

    if prompt_name not in FREE_GENERATION_PROMPT:
        prompt_name = prompt_name[:-2]
    prompt = FREE_GENERATION_PROMPT[prompt_name]
    if tool_name is not None:
        prompt = prompt.replace("[TOOL_NAME]", tool_name)
        prompt = prompt.replace("[TOOL_DOCUMENTATION]", TOOL_DOCUMENTATION[tool_name]["tool_explanation"])
        few_shot_examples = "\n\n".join(TOOL_DOCUMENTATION[tool_name]["few_shot_examples"][few_shot_n:])
        if few_shot_n == 0:
            prompt = prompt.replace("[FEW_SHOT_EXAMPLES]\n\n", "")
        else:
            prompt = prompt.replace("[FEW_SHOT_EXAMPLES]", few_shot_examples)
        prompt = prompt.replace("[ANSWER_EXAMPLE]", TOOL_DOCUMENTATION[tool_name]["example_answer"])
    else:
        if "[TOOL_NAME]" in prompt or "[TOOL_DOCUMENTATION]" in prompt:
            raise Exception("This prompt requires a tool name")
    return prompt



TOOL_SPECS = {
    
"Calculator":{
    "name": "Calculator",
    "arg_parser": lambda x: [calc_parse(x)],
    "tool": Calculator,
    "explanation_prompt": prepare_prompt("ARG_PROMPT", "Calculator"),
    "short_description": "can compute arithmetic expressions",
    "max_arg_length": 30,
    "embedding": torch.tensor([0, 0, 1]),

}, 
"Calendar":{
    "name": "Calendar", 
    "arg_parser": lambda x: [calend_parse(x)],
    "tool": Calendar,
    "explanation_prompt": prepare_prompt("ARG_PROMPT", "Calendar"),
    "short_description": "returns the current date",
    "max_arg_length": 1,
    "embedding": torch.tensor([0, 1, 1]),

}, 
"WikiSearch":{
    "name": "WikiSearch",
    "arg_parser": lambda x: [wiki_parse(x)],
    "tool": WikiSearch,
    "explanation_prompt": prepare_prompt("ARG_PROMPT", "WikiSearch"),
    "short_description": "returns explanations on a subject",
    "max_arg_length": 20,
    "embedding": torch.tensor([1, 0, 1]),
}, 
"GPT3Wiki":{
    "name": "QATool",
    "arg_parser": lambda x: [wiki_parse(x)],
    "tool": GPT3Wiki,
    "explanation_prompt": prepare_prompt("ARG_PROMPT", "GPT3Wiki"),
    "short_description": "replies to general knowledge questions",
    "max_arg_length": 20,
    "embedding": torch.tensor([1, 0, 1]),
},
}


def get_model_path(name):
    global BASE_MODEL_NAME
    return os.path.join("/vol/bitbucket/jg2619/augmenting_llms/model_training/models",f"{BASE_MODEL_NAME}_" + name)

TRAINED_MODELS = {
    "GPTJ-no-add-sub-0": "/vol/bitbucket/jg2619/augmenting_llms/model_training/models/GPTJ_goody",
    "LLAMA-paper": "/vol/bitbucket/jg2619/augmenting_llms/model_training/models/LLAMA_paper",
}




MODEL_NAME = "LLAMA-bare"
BASE_MODEL_NAME = "LLAMA" # "GPTJ" or "LLAMA"

BENCHMARK_NAME = "gsm8k-easy"

#logging

cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache"
def load_LLAMA(path:str="meta-llama/Llama-2-7b-hf",
              new_tokens:List[str]=[],
              **opts):
        from transformers import LlamaTokenizer, LlamaConfig, LlamaForCausalLM
        
        kwargs = {"cache_dir": cache_dir,
                  "token": "***REMOVED***",}
        
        tokenizer = LlamaTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            **kwargs
        )
        tokenizer.add_bos_token = False
        tokenizer.add_tokens(new_tokens)# + ["[PAD]"])
        tokenizer.pad_token = tokenizer.bos_token


        if path == "meta-llama/Llama-2-7b-hf":
            config = LlamaConfig.from_pretrained(
                path, 
                padding_idx=tokenizer.pad_token_id,
                **kwargs
            )
            kwargs["config"] = config
        model = LlamaForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **kwargs
        ).cuda()

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


    output_dir = config.pop("output_dir", ".")
    log_dir = output_dir + "/logs"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if "substitute_explan_tokens" in config:
        config["free_generation_prompt"] = config["free_generation_prompt"].replace("<TOOL>", config["substitute_explan_tokens"][0])
        config["free_generation_prompt"] = config["free_generation_prompt"].replace("</TOOL>", config["substitute_explan_tokens"][1])


    tool_specs = []
    for tool in config.pop("tools"):
        spec = TOOL_SPECS[tool]
        if "override_tool_explan" in config:
            spec["explanation_prompt"] = config["override_tool_explan"][tool]
        if "substitute_explan_tokens" in config:
            spec["explanation_prompt"] = spec["explanation_prompt"].replace("<TOOL>", config["substitute_explan_tokens"][0])
            spec["explanation_prompt"] = spec["explanation_prompt"].replace("</TOOL>", config["substitute_explan_tokens"][1])


        tool_specs.append(spec)
    
    tool_token_ids = config.pop("tool_token_ids", [])
    tool_tokens = config.pop("tool_tokens", "[")
    if tool_token_ids == []:
        for key, value in config["tokenizer"].get_vocab().items():
            if any(token in key for token in tool_tokens):
                tool_token_ids.append(value)

    assert len(tool_tokens) > 0, "No tool tokens found in tokenizer vocab"
    LOGGER.warn(f"Tool tokens: {tool_tokens}")

    

    # Recurse dictionary recursively and print values if they are str, float ot int:
    def print_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                LOGGER.warn('  ' * indent + str(key))
                print_dict(value, indent+4)
            elif isinstance(value, (str, float, int)):
                LOGGER.warn('  ' * indent + str(key) + ": " + str(value))
                
    print_dict(config)
    print("Tool tokens:", tool_tokens)
    print("Tool token ids:", tool_token_ids)
    print("Tool specs:", tool_specs)
    
    return ToolMaster(tool_specs = tool_specs,
                      tool_token_ids = tool_token_ids,
                      log_dir=log_dir,
                      catch_answers=True,
                      answer_token_ids=ANSWER_TOKEN_IDS[BASE_MODEL_NAME],
                      post_answer_token_ids=POST_ANSWER_TOKEN_IDS[BASE_MODEL_NAME],
                      is_llama=True,
                      **config
                      )

@beartype
def create_config(
    base_model_name: str = "LLAMA", # GPTJ or LLAMA
    model_path: str = "meta-llama/Llama-2-7b-hf",
    max_new_tokens: int = 40,
    free_gen_prompt_name: str = "CALC_EXPLAN_1SHOT",
    tools: list[str] = ["Calculator"],
    **kwargs,
):
    global MODEL_NAME, FREE_GENERATION_PROMPT
 
    model, tokenizer = load_LLAMA(model_path, **kwargs)
    max_new_tokens = int(1.2*max_new_tokens)

    config = {
        "model": model,
        "tokenizer": tokenizer,
        "tools": tools,
        "free_generation_prompt": prepare_prompt(free_gen_prompt_name, tools[0]),
        "max_new_tokens": max_new_tokens,
    } | kwargs
 
    return config

def perplexity(dataset):
    # Dataset is a tuple of predictions and labels

    average_perplexity = 0
    examples = 0

    for pred, lab in zip(dataset):
        examples += 1
        loss_fct = torch.nn.functional.cross_entropy(pred.reshape(-1, pred.shape[-1]), lab.view(-1), reduction='sum')

        average_perplexity += torch.exp(loss_fct)

    average_perplexity /= examples
    return {"perplexity": average_perplexity}

# Function that loads and returns the GMS8K dataset
def load_gsm8k_easy():
    with open("ToolQA/data/questions/easy/gsm8k-easy.jsonl", "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    return data

def load_gsm8k_hard():
    with open("ToolQA/data/questions/hard/gsm8k-hard.jsonl", "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    return data


def load_ASDiv(subset = True):

    from bs4 import BeautifulSoup
    import re

    with open('/vol/bitbucket/jg2619/augmenting_llms/benchmarks/ASDiv/ASDiv.xml', 'r') as f:
        data = f.read()
    Bs_data = BeautifulSoup(data, "lxml")

    problem_ids = []

    with open('/vol/bitbucket/jg2619/augmenting_llms/benchmarks/ASDiv/fold0.txt', 'r') as f:
        for line in f.readlines():
            problem_ids.append(line.strip())

    if not subset: problem_ids = [f"nluds-{id:04d}" for id in range(1,2306)]
    data = []

    for id in problem_ids:
        problem = Bs_data.find("problem", id=id)
        # Remove units, provided as " (unit)"
        answer = re.sub(' \(.+\)', '', problem.answer.text)

        question = str(problem.find(string=True, recursive=False)[3:-3] + " " + problem.question.text)

        data.append({"question":question, "answer":answer})

    return data

def load_triviaQA():
    with open('/vol/bitbucket/jg2619/augmenting_llms/benchmarks/TriviaQA/triviaqa-unfiltered/short-unfiltered-web-dev-types.json') as f:
        data = json.load(f)["Data"]
        # Get a random sample of 300 examples, with a fixed seed:
        # data = random.Random(42).sample(data, 900)
        data = [d for d in data if d["answer_type"] == "WikipediaEntity"]
    return data


#def load_ccnet_for_pp():
#val_dataset = load_dataset("/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/data/unprocessed/segment_ccnet_unprocessed", data_files=["1000_examples_not_in_training.csv"], split="train")
#val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)


def perplexity(dataset):
    # Dataset is a tuple of predictions and labels

    average_perplexity = 0
    examples = 0

    for pred, lab in zip(dataset):
        examples += 1
        loss_fct = torch.nn.functional.cross_entropy(pred.reshape(-1, pred.shape[-1]), lab.view(-1), reduction='sum')

        average_perplexity += torch.exp(loss_fct)

    average_perplexity /= examples
    return {"perplexity": average_perplexity}


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
                 batch_size:int = None,):
    
    LOGGER.info("Prompting model.....................................")

    total_time = 0

    model_outputs = []
    pending = questions

    while len(pending) > 0:
        try:
            start_time = time.time()
            model_outputs += model(pending[:batch_size])
        except torch.cuda.OutOfMemoryError as e:             # type: ignore
            LOGGER.info(f"Out of memory error. Reducing batch size from {batch_size} to {batch_size-5}")
            batch_size -= 5
            continue
        except Exception as e:
            print("Exception.... should stop")
            TOOLMASTER_LOGGER.exception(e)
            LOGGER.exception(e)
            raise e

        total_time += time.time() - start_time
        pending = pending[batch_size:]
        
    responses = [output["response"] for output in model_outputs]
    tool_histories = [output["tool_history"] for output in model_outputs]

    # print
    for out in model_outputs:
        LOGGER.info(out)

    return responses, tool_histories, total_time


@beartype
def write_results(
    experiment_data:Union[Dict[List],List[Dict[str, Any]]],
    total_time=None,
    benchmark_name:str = BENCHMARK_NAME,
    model_name:str = MODEL_NAME,
):
    # count csv files in directory
    file_count = len([name for name in os.listdir('./results') if name.endswith(".csv") and model_name in name and benchmark_name in name])

    if isinstance(experiment_data, dict):
        # Transform dict to list of dicts:
        experiment_data = [dict(zip(experiment_data,t)) for t in zip(*experiment_data.values())]

    with open(f"results/{benchmark_name}-responses-{model_name}_{file_count}.csv", "w") as f:
        LOGGER.info(f"Writing responses to {benchmark_name}-responses-{model_name}_{file_count}.csv")
        
        # Standard writer:
        writer = csv.DictWriter(f, fieldnames=list(experiment_data[0].keys()))
        writer.writeheader()
        for row in experiment_data:
            writer.writerow(row)
            for key, value in row.items():
                LOGGER.debug(f"{key}: {value}")
            LOGGER.debug("")

    with open(f"results/{benchmark_name}-responses-{model_name}_{file_count}.txt", "w") as f:
        # Write current date of experiment:
        f.write(f"Date: {datetime.datetime.now()}\n")
        if total_time is not None:
            # Write time taken for experiment:
            f.write(f"Time taken: {total_time}\n")


            
#LOGGER.info("Extracting model answers.....................................")
@beartype
def extract_answer(
        response: str,
) -> str:
    response = response.replace("Answer 2:", "Answer:")
    response = response.replace("Answer 1:", "Answer:")
    response = response.replace("Answer 3:", "Answer:")
    response = response.replace("Answer 4:", "Answer:")
    response = response.replace("Answer 5:", "Answer:")
    # Check that the model has generated a sentence with "Answer(<response>)"
    # Check with re if "answer " or "answer:" is in low_responses:
    if re.search(r"answer", response, flags=re.IGNORECASE) is None:
        #LOGGER.info(f"The model has not generated a sentence with 'Answer'  .......  {'FAIL':>10}   ||  Processed total: {(i+1)*100/len(responses):0.1f}%,  Correct extraction: {correctly_extracted_count/(i+1)}")#logging.info
        return response
    response = re.split(r"answer:?", response, flags=re.IGNORECASE)

    # Add all parts of the response after "Answer" to the answer:
    response = "answer".join(response[1:])

    return response


dot_decimal_not = r'^\.(\d+)'
replacement = r'0.\1'
#LOGGER.info("Extracting first number in answer.....................................")
def first_number(answer:str, equals:bool = True) -> str:
    # We want to extract the first number in each answer:
    # Warn if there is an equal sign in the answer or more than one number
    try:
        numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+', answer)
        if len(numbers) > 1:
            pass
            #LOGGER.debug(f"Warning: More than one number in answer: {answer}")
        first_number = numbers[0]

        first_number = re.sub(dot_decimal_not, replacement, first_number)
        # Equal sign in answer:
        if equals and "=" in answer:
            new_answer = answer.split("=")[1].strip(" $€£\n")
            # LOGGER.warn(f"Warning: Equal sign in answer: {answer}")
            # Get first number after equal sign:
            try:
                new_first_number = re.match(r'[-+]?[0-9]*\.?[0-9]+', new_answer)[0]
                return new_first_number
            except:
                LOGGER.warn(f"Warning: No number after equal sign: {answer}")
                print(f"Warning: No number after equal sign: {answer}")

        return first_number
    except Exception as e:
        return ""
    
def remove_tool_calls(answer:str, tool_name:str=None, start_token:str="<TOOL>", end_token:str="→"):
    if tool_name is not None:
        tool_use_pattern = rf" {re.escape(start_token)}(?={tool_name}).*?{re.escape(end_token)}"
    else:
        tool_use_pattern = rf" {re.escape(start_token)}.*?{re.escape(end_token)}"
    # We want to extract the first number in each answer:
    # Warn if there is an equal sign in the answer or more than one number
    answer = re.sub(tool_use_pattern, "", answer)
    return answer

def trivia_qa_accuracy(
        model_answers:List[str],
        solutions:List[List[str]],
):
    correct_includes = 0

    for correct, model_ans in zip(solutions, model_answers):
        # Get first 20 words of model_ans:
        model_ans = " ".join(model_ans.split()[:20])
        incl_done = False
        for c in correct:
            if not incl_done and str(c) in str(model_ans):
                correct_includes += 1
                incl_done = True

    return correct_includes/len(solutions)

def eval_asdiv(
        responses:List[str],
        solutions:List[str],
):
    results = []
    correct_count = 0
    for resp, sol in zip(responses, solutions, strict=True):
        sol = sol.strip()
        try:
            sol = float(sol)
            # sol is a number so we get first number in response:
            resp = first_number(resp)
            if resp == "":
                results.append(False)
            elif str(float(resp)) == str(sol):
                correct_count += 1
                results.append(True)
            else:
                results.append(False)
        except:
            # sol is not a number. It can be either a "yes" or the name of a person, or a time

            # Check if sol is in first 20 words of response:
            resp = " ".join(resp.split()[:20])
            if str(sol) in resp:
                correct_count += 1
                results.append(True)
            else:
                results.append(False)

    return correct_count/len(solutions), results


def exact_acc(
        list_a:List,
        list_b:List,
        convert:Callable=str,
):
    count = 0
    for a, b in zip(list_a, list_b, strict=True):
        try:
            if convert(a) == convert(b):
                count += 1
        except ValueError:
            pass
        
    
    return count/len(list_a)

def incl_acc(
        list_a:List,
        list_b:List,
        convert:Callable=str,
        det: bool=False
):
    count = 0
    result = []
    for a, b in zip(list_a, list_b, strict=True):
        try:
            if convert(a) in convert(b):
                count += 1
                result.append(True)
            else:
                result.append(False)
        except ValueError:
            result.append(False)
            pass

    if det:
        return count/len(list_a), result

    return count/len(list_a)



@beartype
def experiment(base_model: str,
               model_path:str=None,
               questions:List=None,
               tools: List[str]=["Calculator"],
               batch_size:int = 76,
               exp_ascii:str = "Starting experiment...",
               **kwargs):
    global TOOLMASTER_LOGGER

    config_args = {
        "base_model_name":base_model,
        "model_path":model_path,
        "tools":tools,
    } | kwargs

    if model_path is None:
        del config_args["model_path"]
    config = create_config(**config_args)

    LOGGER.info(f"Config:")
    for key, value in config.items():
        LOGGER.info(f"{key}: {value}")
    LOGGER.info("")
    toolmaster = create_toolformer(config)

    TOOLMASTER_LOGGER = logging.getLogger("Toolmaster")
    TOOLMASTER_LOGGER.warn(exp_ascii)

    if questions is None:
        return toolmaster

    return prompt_model(toolmaster, questions, batch_size=batch_size)


if __name__ == "__main__":
    arg1 = sys.argv[1].replace(",_", ", ")
    arg2 = sys.argv[2].replace(",_", ", ")

    dataset_names = arg1.split(", ")
    experiment_names = arg2.split(", ")
    print(experiment_names)
    print(sys.argv[2])
    print(sys.argv)
    if len(sys.argv) > 2:
        project_description = sys.argv[3]
    else:
        project_description = "default"
        
    FILE_HANDLER = logging.FileHandler(f"{eval_logs_dir}/{project_description}.log")
    FILE_HANDLER.setLevel(logging.INFO)
    FILE_HANDLER.setFormatter(formatter)

    DEBUG_HANDLER = logging.FileHandler(f"{eval_logs_debug_dir}/{project_description}.log")
    DEBUG_HANDLER.setLevel(logging.DEBUG)
    DEBUG_HANDLER.setFormatter(formatter)

    LOGGER.addHandler(STREAM_HANDLER)
    LOGGER.addHandler(FILE_HANDLER)
    LOGGER.addHandler(DEBUG_HANDLER)

    datasets = []
    # LOAD DATA:
    for dname in dataset_names:
        match dname.lower():
            case "gsm8k-easy":
                data = {"data":load_gsm8k_easy(),
                        "type":"math"}
            case "gsm8k-hard":
                data = {"data":load_gsm8k_hard(),
                        "type":"math"}
            case "asdiv":
                data = {"data":load_ASDiv(),
                        "type":"math",
                        "config_kwargs":{"debug_level": 1}}
            case "asdiv-full":
                data = {"data":load_ASDiv(False),
                        "type":"math",
                        "config_kwargs":{"debug_level": 1}}
            case "triviaqa":
                data = {"data":load_triviaQA(),
                        "type":"wiki",}
            case "triviaqa-small":
                data = {"data":load_triviaQA()[:500],
                        "type":"wiki",}
            case "test":
                data = {"data":load_ASDiv()[:10],
                        "type":"math",
                        "config_kwargs":{"max_new_tokens": 40,
                                         "debug_level": 2}}
            case _:
                raise ValueError(f"Dataset name {dname.lower()} not recognised")
            
        data["name"] = dname
        datasets.append(data)

    for name in experiment_names:
        for dataset in datasets:

            exp_ascii = f"""
###########################################################################\n
    * Running {name} experiment on dataset {dataset['name']} *  

    * Experiment id: {project_description} *\n
###########################################################################\n\n"""

            

            LOGGER.info(exp_ascii)

            
            ex_config = {
                "max_new_tokens": 40,
                "experiment_name": project_description,
            }
            extract_answers = False

            # Can be overriden by experiments:
            if dataset["type"] == "math":
                ex_config["tools"] = ["Calculator"]
            if dataset["type"] == "wiki":
                ex_config["tools"] = ["WikiSearch"]
            
            ex_config = ex_config | dataset.get("config_kwargs", {})

            if name.startswith("custom"):
                # Model path - tokentype - tool explan mode - task explan mode - few shot n
                config = name.split("_")
                new_configs = []
                for i, c in enumerate(config):
                    if c.startswith("["):
                        c.strip("[]")
                        c = c.split(",")
                        for ci in c:
                            new_configs.append("_".join(config[:i] + [ci] + config[i+1:]))
                        break
                if len(new_configs) > 0:
                    experiment_names.append(new_configs)
                    continue
                    
                
                model_path = config[1]
                tool_explan_mode = config[2] if len(config) > 2 else 0
                task_explan_mode = config[3] if len(config) > 3 else 0
                few_shot_n = int(config[4]) if len(config) > 4 else 0
                tokentype = config[5] if len(config) > 5 else 1

                ex_config["base_model"] = "GPTJ"
                try:
                    ex_config["model_path"] = TRAINED_MODELS[model_path]
                except KeyError:
                    ex_config["model_path"] = get_model_path(model_path)
                if tokentype == 1:
                    ex_config["new_tokens"] = [" <TOOL>", "</TOOL>"]
                    ex_config["tool_tokens"] = [" <TOOL>"]
                    ex_config["end_tool_token"] = "</TOOL>"
                    if tool_explan_mode != 1:
                        ex_config["substitute_explan_tokens"] = ["<TOOL>", "</TOOL>"]

                ex_config["free_gen_prompt_name"] = f"{tool_explan_mode} {task_explan_mode} {few_shot_n}"

                if tool_explan_mode == 2:
                    ex_config["pretty_tools"] = True

                if tool_explan_mode == 1 or few_shot_n > 0:
                    catch_answers = True
               
            else:
                match name:
                    case "baseline":
                        ex_config["base_model"] = "LLAMA"
                        ex_config["disable_tools"] = True
                        ex_config["free_gen_prompt_name"] = "0 1 spec"
                    case "llama-1":
                        ex_config["base_model"] = "LLAMA"
                        ex_config["model_path"] = TRAINED_MODELS["LLAMA-paper"]
                        ex_config["new_tokens"] = ["<TOOL>", "</TOOL>"]
                        ex_config["tool_tokens"] = ["<TOOL>"]
                        ex_config["end_tool_token"] = "</TOOL>"
                        ex_config["free_gen_prompt_name"] = "2 3 spec"
                        ex_config["substitute_explan_tokens"] = ["<TOOL>", "</TOOL>"]
                        #ex_config["override_tool_explan"] = {tool:prepare_prompt("0 1", tool) for tool in ex_config["tools"]}
                        #ex_config["debug_level"] = 2
                        ex_config["pretty_tools"] = True
                        ex_config["tool_top_k"] = 10
                    case "LLAMA_baseline_0.5":
                        ex_config["base_model"] = "LLAMA"
                        ex_config["disable_tools"] = True
                        ex_config["free_gen_prompt_name"] = "0.5 spec"
                    case "LLAMA_Master_1.5":
                        ex_config["base_model"] = "LLAMA"
                        ex_config["free_gen_prompt_name"] = "1.5 spec"
                        ex_config["substitute_explan_tokens"] = ["[", "]"]

                    case "LLAMA_baselineb":
                        ex_config["base_model"] = "LLAMA"
                        ex_config["disable_tools"] = True
                        ex_config["free_gen_prompt_name"] = "0 1b spec"
                    case "LLAMA_Masterb":
                        ex_config["base_model"] = "LLAMA"
                        ex_config["free_gen_prompt_name"] = "1 1b spec"
                        ex_config["substitute_explan_tokens"] = ["[", "]"]

                    case "LLAMA_baseline":
                        ex_config["base_model"] = "LLAMA"
                        ex_config["disable_tools"] = True
                        ex_config["free_gen_prompt_name"] = "0 1 spec"
                    case "LLAMA_baseline+":
                        ex_config["base_model"] = "LLAMA"
                        ex_config["disable_tools"] = True
                        ex_config["free_gen_prompt_name"] = "0 1+ spec"
                    case "LLAMA_Master":
                        ex_config["base_model"] = "LLAMA"
                        ex_config["free_gen_prompt_name"] = "1 1 spec"
                        ex_config["substitute_explan_tokens"] = ["[", "]"]
                    case "LLAMA_Master-0shot":
                        ex_config["base_model"] = "LLAMA"
                        ex_config["free_gen_prompt_name"] = "1.5 spec"
                        ex_config["substitute_explan_tokens"] = ["[", "]"]

                        TOOL_DOCUMENTATION["Calculator"]["tool_explanation"] = """The Calculator tool computes arithmetic expressions. You can call the API by writing "[Calculator(expression→]" where "expression" is the expression to be computed."""
                        TOOL_DOCUMENTATION["WikiSearch"]["tool_explanation"] = """The WikiSearch tool retrives Wikipedia snipets. You can use it to look up encyclopedic information from the current context. You can do so by writing "[WikiSearch(term)→]" where "term" is the search term you want to look up."""
                        
                        ex_config["top_k"] = 50
                    case "LLAMA_Master-2shot-5":
                        ex_config["base_model"] = "LLAMA"
                        ex_config["free_gen_prompt_name"] = "1 2 spec"
                        ex_config["substitute_explan_tokens"] = ["[", "]"]

                        TOOL_DOCUMENTATION["Calculator"]["tool_explanation"] = """The Calculator tool computes arithmetic expressions. You can call the API by writing "[Calculator(expression→]" where "expression" is the expression to be computed."""
                        TOOL_DOCUMENTATION["WikiSearch"]["tool_explanation"] = """The WikiSearch tool retrives Wikipedia snipets. You can use it to look up encyclopedic information from the current context. You can do so by writing "[WikiSearch(term)→]" where "term" is the search term you want to look up."""
                        
                        ex_config["top_k"] = 5
                    case "LLAMA_Master-2shot-15":
                        ex_config["base_model"] = "LLAMA"
                        ex_config["free_gen_prompt_name"] = "1 2 spec"
                        ex_config["substitute_explan_tokens"] = ["[", "]"]

                        TOOL_DOCUMENTATION["Calculator"]["tool_explanation"] = """The Calculator tool computes arithmetic expressions. You can call the API by writing "[Calculator(expression→]" where "expression" is the expression to be computed."""
                        TOOL_DOCUMENTATION["WikiSearch"]["tool_explanation"] = """The WikiSearch tool retrives Wikipedia snipets. You can use it to look up encyclopedic information from the current context. You can do so by writing "[WikiSearch(term)→]" where "term" is the search term you want to look up."""
                        
                        ex_config["top_k"] = 15


                    case _:
                        raise ValueError(f"Experiment name {name} not recognised")



                if "spec" in ex_config["free_gen_prompt_name"]:
                    ex_config["free_gen_prompt_name"] = ex_config["free_gen_prompt_name"].replace("spec", dataset["type"])



            
            # TODO BASELINE GPTJ
            # TriviaQA
            # SVAMP
            # Wiki experiments

            #print(data)
            questions = [d["question"] for d in dataset["data"]]

            # Print ex config in pretty format:
            print(f"EXPERIMENT CONFIGURATION:\n{json.dumps(ex_config, indent=4)}")

            responses, tool_histories, total_time = experiment(
                questions=questions,
                exp_ascii=exp_ascii,
                **ex_config
            )


            ######################### EVALUATION #########################

            metrics_to_report = []

            remove_calls_arr = partial(remove_tool_calls, start_token=ex_config.get("substitute_explan_tokens",["<TOOL>"])[0], end_token="→")
            remove_calls_end = partial(remove_tool_calls, start_token=ex_config.get("substitute_explan_tokens",["<TOOL>"])[0], end_token=ex_config.get("substitute_explan_tokens",["","</TOOL>"])[1])
            extracted_answers = list(map(extract_answer, responses))
            call_less_arr_responses = list(map(remove_calls_arr, responses))
            call_less_end_responses = list(map(remove_calls_end, responses))
            extracted_cla_responses = list(map(extract_answer, call_less_arr_responses))
            extracted_cle_responses = list(map(extract_answer, call_less_end_responses))

            correct_answers = [d["answer"] for d in dataset["data"]]

            ex_results = {"questions":questions,"correct_answers":correct_answers,"responses":responses, "extracted":extracted_answers, "tool_histories":tool_histories}

            generous_acc, result = incl_acc(correct_answers, responses, det = True)
            metrics_to_report.append(f"Generous accuracy: {generous_acc}")

            results_asdiv_exa = []
            if dataset["type"] == "math":
                first_numbers_arr = list(map(first_number, call_less_arr_responses))
                first_numbers_end = list(map(first_number, call_less_end_responses))
                first_numbers_cla = list(map(first_number, extracted_cla_responses))
                first_numbers_cle = list(map(first_number, extracted_cle_responses))

                print(f"FIRST NUMBERS ARR: {first_numbers_arr}")

                ex_results["first_numbers_arr"] = first_numbers_arr
                ex_results["first_numbers_end"] = first_numbers_end

                fn_arr_acc = exact_acc(first_numbers_arr, correct_answers, float)
                fn_end_acc = exact_acc(first_numbers_end, correct_answers, float)
                fn_cla_acc = exact_acc(first_numbers_cla, correct_answers, float)
                fn_cle_acc = exact_acc(first_numbers_cle, correct_answers, float)

                asdiv_acc_a, results_asdiv_a = eval_asdiv(call_less_arr_responses, correct_answers)
                asdiv_acc_e, results_asdiv_e = eval_asdiv(call_less_end_responses, correct_answers)
                asdiv_acc_exa, results_asdiv_exa = eval_asdiv(extracted_cla_responses, correct_answers)
                asdiv_acc_exe, results_asdiv_exe = eval_asdiv(extracted_cle_responses, correct_answers)

                metrics_to_report.append(f"First number in answer (arr): {fn_arr_acc}")
                metrics_to_report.append(f"First number in answer (end): {fn_end_acc}")
                metrics_to_report.append(f"First number in answer (extracted arr): {fn_cla_acc}")
                metrics_to_report.append(f"First number in answer (extracted end): {fn_cle_acc}")

                metrics_to_report.append(f"ASDiv accuracy (arr): {asdiv_acc_a}")
                metrics_to_report.append(f"ASDiv accuracy (end): {asdiv_acc_e}")
                metrics_to_report.append(f"ASDiv accuracy (extracted arr): {asdiv_acc_exa}")
                metrics_to_report.append(f"ASDiv accuracy (extracted end): {asdiv_acc_exe}")

                print(f"ASDIV RESULTS ARR: {results_asdiv_a}")
                print(f"ASDIV RESULTS END: {results_asdiv_e}")
                print(f"ASDIV RESULTS EXTRACTED ARR: {results_asdiv_exa}")
                print(f"ASDIV RESULTS EXTRACTED END: {results_asdiv_exe}")

                print(f"Sum of asdiv results ex arr: {sum(results_asdiv_exa)}")
                print(f"Len of asdiv results ex arr: {len(results_asdiv_exa)}")
                print(f"Acc of asdiv results ex arr: {sum(results_asdiv_exa)/len(results_asdiv_exa)}")

                assert len(results_asdiv_a) == len(results_asdiv_e) == len(results_asdiv_exa) == len(results_asdiv_exe) == len(correct_answers)


            if "triviaqa" in dataset["name"].lower():
                ex_results["answer_aliases"] = [d["answer_aliases"] for d in dataset["data"]]
                trivia_first20 = trivia_qa_accuracy(call_less_end_responses, [d["answer_aliases"] for d in dataset["data"]])
                trivia_first20_extr = trivia_qa_accuracy(extracted_cle_responses, [d["answer_aliases"] for d in dataset["data"]])
                metrics_to_report.append(f"Trivia top 20: {trivia_first20}")
                metrics_to_report.append(f"Trivia top 20 (answer extracted): {trivia_first20_extr}")


            
            #stats(model_answers=extracted_answers, correct_answers=correct_answers, tool_history=tool_histories, answer_type="extracted")
            #stats(result, tool_history=tool_histories, answer_type="full-responses")
            
            metrics = "\n".join(metrics_to_report)
            results_ascii = f"""
###########################################################################\n
    * FINISHED {name} experiment on dataset {dataset['name']} *  

    * Experiment id: {project_description} *

    * Total time taken: {total_time} *

    [metrics]

###########################################################################\n\n""".replace("[metrics]", metrics)
            
            LOGGER.info(f"Finished experiment {name} for {dataset['name']}")
            LOGGER.warn(results_ascii)
            TOOLMASTER_LOGGER.warn(results_ascii)
            LOGGER.debug(ex_results)
            write_results(ex_results, total_time, benchmark_name = dataset["name"], model_name = f"{project_description}_{name}")

            
            for i, (question, resp, answer) in enumerate(zip(questions, responses, correct_answers, strict=True)):
                print("________________________________________________________________________________")
                print(f"QUESTION: {question}")
                if len(results_asdiv_exa) == len(question): print(f"RESULT (arr): {results_asdiv_exa[i]}, (end): {results_asdiv_exe[i]}")
                print(f"ANSWER: {answer}")
                print(f"RESPONSE: {resp}".replace("\n", "\n--"))
                print("________________________________________________________________________________")
                print()

            torch.cuda.empty_cache()

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
