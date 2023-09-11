import os
import logging
import re
import sys
import traceback

from functools import partial

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import tensor as Tensor

import numpy as np
import openai

from beartype import beartype
from beartype.typing import List, Callable, Union, Dict, Tuple

# Import parent class of AutoTokenizer

from importlib import reload

formatter = logging.Formatter('%(asctime)s:  %(message)s', datefmt='%m/%d/%Y %I:%M:%S  ')

STREAM_HANDLER = logging.StreamHandler(sys.stdout)
STREAM_HANDLER.setLevel(logging.WARN)
STREAM_HANDLER.setFormatter(formatter)

log_dir = "./toolmaster_logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

FILE_HANDLER = logging.FileHandler(f"{log_dir}/default.log")
FILE_HANDLER.setLevel(logging.INFO)
FILE_HANDLER.setFormatter(formatter)

LOGGER = logging.getLogger("Toolmaster")
LOGGER.setLevel(logging.DEBUG)

# Set the output file for the logger:


pad_sequence = partial(pad_sequence, batch_first=True)
longTensor = partial(Tensor, dtype=torch.long)

PAD_TOKEN = "[PAD]"
PAD_ID = -100
ARROW_TOKEN = 39310
TOOL_TOKEN = "["
END_TOOL_TOKEN = "]"
TOOL_TOKEN_IDS = []
END_API_TOKEN = 50401
OPEN_PARENTHESIS = "|"
OPEN_PARENTHESIS_ID = 91
CLOSE_PARENTHESIS = 8

GPTJ_ARG_STOPPERS = [39310, 15168, 15437, 25295, 48600, 
                     35944,  #  ]).
                     5974,   #  ]:
                     46570,  #  ]),
                     12962,  #  ])
                     45297,  #  ]-
                     16151,  #  ](
                     22241,  #  ]=
                     30866,  #  ]"
                     48688,  #  ]+
                     60,     #  ]
                     4083,   #  ].
                     4357,   #  ],
                     91,     #  |
                     ]

GPTJ_BAR_TOKEN = 91

ROW_STATUS_MEANING = {
     0: "Selecting tool",
     1: "Normal generation",
     2: "Normal generation - waiting to catch answer",
    -1: "Done, normal termination",
    -2: "Max tokens reached",
    -3: "Answer caught - stopping generation"
}

TOOL_USE_STATUS_MEANING = {
    0: "Correct use",
    1: "Max tokens reached before execution",
    2: "Failed to generate args",
    3: "Use error",
}

LOGIT_DISPLACEMENT = 0 # This is for models where model at position i gives logits of prediction AFTER seeing i. For models that give logits of prediction BEFORE seeing i, this should be 1.


def log(t, eps=1e-20): return t.clamp(min=eps).log()


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1., dim=-1, eps=1e-10):
    # Returns flat vector
    if temperature == 0:
        return t.argmax(dim=dim)

    return ((t / max(temperature, eps)) + gumbel_noise(t)).argmax(dim=dim)

def renumerate(sequence, start=None):
    n = start
    if start is None:
        n = len(sequence) - 1
    for elem in sequence[::-1]:
        yield n, elem
        n -= 1

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(text, model="text-embedding-ada-002"):
   # Return random vector of length 3:
   return np.random.rand(3)
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']


# Tools given to the toolmaster must have:
# 1. Name: str - Unique identifier for the tool
# 2. Arg parser: Callable - A function that takes a string and returns a list of arguments
# 3. Tool: Callable - A function that takes a list of argumets and returns a string
# 4. Explanation prompt: Union[torch.Tensor, str] - A string that explains how to use the tool
# 5. Short description: Optional[str] - A short description of the tool
# 6. Embedding: Optional[torch.Tensor] - A vector that represents the tool. Necessary for method b.


@beartype
class ToolMaster(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        tool_specs: List[Dict],  # of the form {"name": str, "arg_parser": Callable, "tool": Callable, "explanation_prompt": Union[torch.Tensor, str], "short_desc": Optional[str], "max_arg_tokens": int, "embedding": Optional[torch.Tensor]}
        tokenizer,
        tool_token_ids: List[int],
        end_tool_token: str = END_TOOL_TOKEN,
        free_generation_prompt: str = None,
        debug_level: int = 0,
        log_dir: str = "/vol/bitbucket/jg2619/augmenting_llms/model_training/models/logs",
        export_tool_execution: bool = False,
        max_new_tokens: int = 30,
        max_response_len: int = 100,
        max_exec_tokens: int = 140,
        temperature: float = 0.1,
        catch_answers: bool = False,
        answer_token_ids: List[int] = None,
        post_answer_token_ids: List[int] = None,
        disable_tools:bool = False,
        pretty_tools: bool = False,
        free_generation_batch_size: int = 251,
        arg_sampling_batch_size: int = 251,
        greedy_sampling: bool = True,
        tool_top_k: int = 1,
        method_b: bool = False,
        max_intention_tokens: int = 25,
        **kwargs,
    ):
        super().__init__()

        global PAD_ID, PAD_TOKEN, OPEN_PARENTHESIS_ID, TOOL_TOKEN_IDS, LOGGER, STREAM_HANDLER, END_TOOL_TOKEN
        
        self.model = model
        self.device = model.device
        self.encode = tokenizer.encode
        self.decode = tokenizer.decode

        PAD_ID = tokenizer.pad_token_id
        PAD_TOKEN = tokenizer.pad_token
        TOOL_TOKEN_IDS = longTensor(tool_token_ids, device=self.device).view(-1)

        OPEN_PARENTHESIS_ID = tokenizer.encode(OPEN_PARENTHESIS)
        assert len(OPEN_PARENTHESIS_ID) == 1, "Open parenthesis token must be a single token"
        OPEN_PARENTHESIS_ID = longTensor(OPEN_PARENTHESIS_ID).to(self.device)

        tool_names = [tool_spec["name"] for tool_spec in tool_specs]
        tokenized_tools = [tokenizer.encode(tool_name) for tool_name in tool_names]
        self.tokenized_tools = tokenized_tools
        self.tool_names = tool_names

        self.free_gen_batch_size = free_generation_batch_size
        self.arg_sampling_batch_size = arg_sampling_batch_size

        tool_name_desc = []
        for spec in tool_specs:
            name_desc = spec["name"]
            if 'short_description' in spec:
                name_desc += " (" + spec['short_description'] +")"
            tool_name_desc.append(name_desc)
        if pretty_tools:
            tool_name_desc = [f"  - {name_desc}" for name_desc in tool_name_desc]
            tool_name_desc = "\n".join(tool_name_desc)
        else:
            tool_name_desc = ", ".join(tool_name_desc)

        if free_generation_prompt is None:
            free_generation_prompt = "You can use these tools to help you answer: [AVAILABLE TOOLS].\n\n"
        
        free_generation_prompt = free_generation_prompt.replace("[AVAILABLE TOOLS]", tool_name_desc)
        self.free_generation_prompt = free_generation_prompt.replace("[PROMPT]", "")
        
        if "[PROMPT]" in free_generation_prompt:
            self.free_gen_sub_idx = len(self.encode(free_generation_prompt.split("[PROMPT]")[0]))
        else:
            self.free_gen_sub_idx = len(self.encode(free_generation_prompt))
        self.tokenized_free_generation_prompt = longTensor(self.encode(self.free_generation_prompt)).to(self.device)

        tool_selection_dict = {}
        # This function creates a decision tree for the tool selection. The model chooses at each depth the token with the highest probability, until it reaches a tool id.
        def tree_maker(tree, token, id, depth):
            tokens = list(tree.keys())
            if token not in tokens:
                tree[token] = id 
            else:
                if token == OPEN_PARENTHESIS_ID:
                    print(f"Warning: tool {tokenized_tools[id]} is already in the tree")
                    LOGGER.warning(f"Warning: tool {tokenized_tools[id]} is already in the tree")
                    return
                # Check if instance of dictionary:
                if not isinstance(tree[token], dict):
                    other_id = tree[token]
                    next_token = tokenized_tools[other_id][depth+1] if depth + 1 < len(tokenized_tools[other_id]) else OPEN_PARENTHESIS_ID
                    tree[token] = {next_token: other_id}
                next_token = tokenized_tools[id][depth+1] if depth + 1 < len(tokenized_tools[id]) else OPEN_PARENTHESIS_ID
                tree_maker(tree[token], next_token, id, depth + 1)

        for i, tool in enumerate(tokenized_tools):
            tree_maker(tool_selection_dict, tool[0], i, 0)

        END_TOOL_TOKEN = end_tool_token

        self.tool_selection_dict = tool_selection_dict
        tool_explanation_prompts = [tool_spec["explanation_prompt"] for tool_spec in tool_specs]
        prepare_explan_for_gen = lambda x: longTensor(self.encode(x.replace("[PROMPT]", ""))).to(self.device)
        self.tool_explanation_prompts = list(map(prepare_explan_for_gen,  tool_explanation_prompts))
        self.disable_tools = disable_tools
        self.greedy_sampling = greedy_sampling
        self.tool_top_k = tool_top_k

        self.close_bad_arg = Tensor(self.encode(f"{PAD_TOKEN}{END_TOOL_TOKEN}")[1:]).to(self.device)

        self.tool_explan_sub_indices = []
        for explan in tool_explanation_prompts:
            if "[PROMPT]" in explan:
                self.tool_explan_sub_indices.append(len(self.encode(explan.split("[PROMPT]")[0])))
            else:
                self.tool_explan_sub_indices.append(len(self.encode(explan)))
    
        self.tools = [tool_spec["tool"] for tool_spec in tool_specs]
        self.arg_parsers = [tool_spec["arg_parser"] for tool_spec in tool_specs]
        self.max_arg_lengths = [tool_spec["max_arg_length"] for tool_spec in tool_specs]
        self.tokenized_tools = tokenized_tools

        self.max_new_tokens = max_new_tokens
        self.max_exec_tokens = max_exec_tokens
        self.max_response_len = max_response_len
        self.temperature = temperature

        self.debug_level = debug_level
        # Create log dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # count files in log dir
        i = len(os.listdir(log_dir))

        FILE_HANDLER.setLevel(logging.DEBUG if debug_level>0 else logging.INFO)
        STREAM_HANDLER.setLevel(logging.DEBUG if debug_level>1 else logging.INFO)
        LOGGER.addHandler(STREAM_HANDLER)
        LOGGER.addHandler(FILE_HANDLER)

        self.catch_answers = catch_answers
        self.answer_token_ids = torch.tensor(answer_token_ids, dtype=torch.int32, device=self.device)
        if post_answer_token_ids is not None:
            post_answer_token_ids = longTensor(post_answer_token_ids).to(self.device)
        self.post_answer_token_ids = post_answer_token_ids

        # Tokens with →
        self.arg_gen_stoppers = []
        for key, value in tokenizer.get_vocab().items():
            if "→" in key or "]" in key or "|" in key:     # or ")" in key:  # Remove close parenthesis TODO
                self.arg_gen_stoppers.append(value)
        self.arg_gen_stoppers = GPTJ_ARG_STOPPERS
        self.arg_gen_stoppers = Tensor(self.arg_gen_stoppers).to(self.model.device)

        self.export_tool_execution = export_tool_execution

        self.method_b = method_b
        self.max_intention_tokens = max_intention_tokens
        self.bar_token = tokenizer.encode("|")[-1]
        

        # COPY PROMPT X TIMES
        # self.tokenized_free_generation_prompt.unsqueeze(0).repeat(batch_size,1)

        for key, value in kwargs.items():
            LOGGER.warn(f"Waring: {key} with value {value} is not a valid argument for the model.")

        LOGGER.info(f"Initialized ToolMaster model with {len(tokenized_tools)} tools")

    @torch.no_grad()
    def generate(self, 
                 user_prompts: list[torch.Tensor], 
                 explanation_prompts: Union[List[torch.Tensor], torch.Tensor],
                 tool_use_len: List[int],
                 tool_histories: List[List[Dict]] = None,
                 arg_selection_mode: bool = False,  # Arg selection mode VS free generation augmented with tool selection
                 max_new_tokens: Union[torch.Tensor,int] = 100, # Max length per sequence or for all sequences
                 temperature: float = 0.8, 
                 stop_tokens: Union[List[int],int,torch.Tensor] = []):
        
        global PAD_ID, PAD_TOKEN, OPEN_PARENTHESIS_ID, TOOL_TOKEN_IDS, LOGIT_DISPLACEMENT
        device = self.device

        # Each data point as it goes through the loop should have:
        # 1. An updated prime: The data point to be completed including the generated content
        # 2. The original prime: The prompt that was prepended to the prime
        # 3. Total generated content: The generated content for the data point
        # 4. Current generated content: The last generated content for the data point in the current loop section
        # 5. The tool history

        if not isinstance(explanation_prompts, list):
            explanation_prompts = [explanation_prompts for _ in range(len(user_prompts))]

        assert user_prompts[0].dim() == 1, "Primes must be 1D tensor with the tokenized data"

        assert len(explanation_prompts) == len(user_prompts), f"If explanations is a list, it must have the same length as the primes. \nExplanations: {explanation_prompts}\nPrimes: {user_prompts}"
        assert explanation_prompts[0].dim() == 1, "Explanation prompts must be 1D tensors"
         
        # Device assertions
        assert user_prompts[0].device == device, "Primes must be on the same device as the model"
        assert explanation_prompts[0].device == device, "Prompts must be on the same device as the model"
        
        batch_size = len(user_prompts)                                                  # BATCH SIZE
        explan_prompt_lens = Tensor([prompt.shape[0] for prompt in explanation_prompts]).to(device)  # LENGTHS OF PREPENDED PROMPTS
        
        if tool_histories is None:        # History of tools used for each row
            tool_histories = [[] for _ in range(batch_size)]
        
        if isinstance(stop_tokens, int):
            stop_tokens = [stop_tokens]
        if not isinstance(stop_tokens, torch.Tensor):
            stop_tokens = longTensor(stop_tokens).to(device).view(-1)
        if isinstance(max_new_tokens, int):
            max_new_tokens = Tensor([max_new_tokens for _ in range(batch_size)]).to(device)

        # Position of where to start generating for each row
        # positions = Tensor([prompt.shape[0]-1 for prompt in user_prompts]).to(device).unsqueeze(1) + explan_prompt_lens.unsqueeze(1)
        # initial_positions = positions.clone() + 1 # Initial positions of each row for data + prompt

        status = Tensor([1 for _ in range(batch_size)]).int().to(device) # Status of each row
        if self.catch_answers and not arg_selection_mode:
            status *= 2
            stop_tokens = self.post_answer_token_ids
        caught = torch.tensor([torch.isin(prime, self.answer_token_ids).any() for prime in user_prompts]).to(device)
        status[caught] = 1
        # 0: selecting tool
        # 1: Generating      (Can be stopped by stop tokens)
        # 2: Generating - catching answer (Waiting to see Ans before being able to catch stop tokens)
        # -1: Done
        # -2: Max tokens     (remove gen content if arg selecting)
        # -3: Answer caught

        # DELETE generated_content_count. positions, initial_positions

        joined_prompts = [torch.cat((explanation_prompts[i], user_prompts[i])) for i in range(batch_size)] # Joined prompts for each row
        reversed_joined_prompts = [prompt.flip(dims=(-1,)) for prompt in joined_prompts]
        left_pad_prompts = pad_sequence(reversed_joined_prompts, padding_value=PAD_ID)
        batch_input = left_pad_prompts.flip(dims=(-1,))
        initial_prompt_length = batch_input.shape[1]
        attention_mask = batch_input != PAD_ID
        new_generated_content = [None for _ in range(batch_size)]
        intention_tokens = [[] for _ in range(batch_size)]
        
        # extra_pad = (batch_lengths + max_new_tokens - batch_generated_count - batch_input.shape[1]).max().item()
        # batch_input = F.pad(batch_input, (0, extra_pad,), value=PAD_ID)

        # LOGGER.debug(f"Extra pad: {extra_pad}")
        #LOGGER.debug(f"Batch lengths: {batch_lengths}")
        LOGGER.debug(f"Max new tokens: {max_new_tokens}")
        LOGGER.debug(f"Batch input shape: {batch_input.shape}")
        LOGGER.debug("Primes:")
        for i, row in enumerate(batch_input):
            LOGGER.debug(self.decode(row))
            #LOGGER.debug(f"Position: {positions[i].item()}, token_id: {row[positions[i].item()].item()}, token: {self.decode(row[positions[i].item()].item())}")
            #LOGGER.debug(f"Position + 1: {positions[i].item()+1}, token_id: {row[positions[i].item()+1].item()}, token: {self.decode(row[positions[i].item()+1].item())}")

        LOGGER.debug(f"Stop tokens: {stop_tokens}")
        
        # Indexing tensor utils
        loop_to_data_idx = torch.arange(batch_size).to(device)                 # Mapping from loop index to batch index
        batch_indices = torch.arange(batch_size).to(device).unsqueeze(1)       # ARANGE THAT ADJUSTS TO THE LOOP BATCH SIZE AS SAMPLES FINISH
        
        # Tool selection utils
        loop_selection_depth = torch.zeros(batch_size).int().to(device)        # Depth of the tool selection tree
        current_opts = [self.tool_selection_dict for _ in range(batch_size)]   # Current tool options for row
        tools_are_disabled = self.disable_tools or arg_selection_mode

        return_list = [None for _ in range(batch_size)]                        # List of return values for each row
        
        past_key_values = None
        while batch_indices.shape[0] > 0:                                                                
            # Logits shape is b*seq length on the first iter and then its b*1
            # Remove assertion: TODO
            assert loop_to_data_idx.shape[0] == batch_indices.shape[0], "Loop to data index and batch indices must have the same shape"

            # MODEL FORWARD CALL. MAINTAINS SHAPE EVEN AFTER INDEXING
            input = self.model.prepare_inputs_for_generation(batch_input, attention_mask=attention_mask, past_key_values = past_key_values, use_cache=True)
            output = self.model(**input)

            loop_last_logits = output.logits[:, -1, :]
            past_key_values = output.past_key_values
            output = None
            # Check if past key values in output:
            #loop_last_logits[:, :, TOOL_TOKEN_IDS[0]] += 10   #TOOL_TOKEN_ID , 13 fulstp

            if tools_are_disabled:   # Tool usage not available
                loop_last_logits[:, TOOL_TOKEN_IDS] = -1e10

            # Gumbel sample for rows not selecting a tool. Tool selection has different sampling procedure
            loop_sampled = torch.ones(batch_indices.shape[0], 1).long().to(device)*-1
            if self.greedy_sampling:
                samp = loop_last_logits[status>0].argmax(dim=-1).unsqueeze(1)
            else:
                samp = gumbel_sample(loop_last_logits[status>0], temperature=temperature).unsqueeze(1)
            # Get the self.tool_top_k top k logits
            if self.tool_top_k > 1 and samp.shape[0] > 0:
                _, top_k = loop_last_logits[status>0].topk(self.tool_top_k, dim=-1)
                # If the top k index , insert tool token in that position
                present = torch.tensor([torch.isin(TOOL_TOKEN_IDS[0], top_k[i], assume_unique=True) for i in range(samp.shape[0])]).to(device)
                samp[present] = TOOL_TOKEN_IDS[0]
            loop_sampled[status>0] = samp

            # Catch answers     2 -> 1
            if not arg_selection_mode and self.catch_answers:
                # Check if any of the tokens are answer tokens
                caught_answers = torch.isin(loop_sampled, self.answer_token_ids).squeeze(1).bool()
                status[caught_answers] = 1   # Generating status (can be stopped)

            # Sampling procedure for rows selecting a tool
            if self.method_b:
                loop_sampled[status==0] = gumbel_sample(loop_last_logits[status==0], temperature=temperature).unsqueeze(1)
                finished_intent = torch.isin(loop_sampled, self.bar_token).squeeze(1) & (status==0)
                status[finished_intent] = -1 if max_new_tokens[selecting_i] > 0 else -2

                # Add token sampled for eaach status==0 to intention tokens:
                for i in (status==0).nonzero().squeeze(1):
                    intention_tokens[i].append(loop_sampled[i].item())
            else:
                # 0 -> -1, -2
                for selecting_i in (status==0).nonzero().squeeze(1):
                        # data_i = loop_to_data_idx[selecting_i].item()
                        # Tool names are composed of tokens. ie. [CAL] [CUL] [ATOR]. We call each token a syllable
                        # Options for the next syllable. 
                    syllable_opts = Tensor(list(current_opts[selecting_i].keys())).to(device)
                    next_syllable_idx = loop_last_logits[selecting_i,syllable_opts].argmax(dim=-1)
                    next_syllable = syllable_opts[next_syllable_idx].item() 
                    loop_sampled[selecting_i] = next_syllable
                    loop_selection_depth[selecting_i] += 1
                    current_opts[selecting_i] = current_opts[selecting_i][next_syllable]

                    # IF current opts is a dict, there is a tie between possible tools. We need to keep selecting syllables.
                    # PASS
                    if not isinstance(current_opts[selecting_i], dict):
                        # ELSE: We've reached a tool id
                        tool_id = current_opts[selecting_i]
                        depth = loop_selection_depth[selecting_i].item()-1      # Selection_depth = i means we've selected the ith syllable of tool name. -1 for indexing purposes.
                        tool_len = len(self.tokenized_tools[tool_id])

                        # tokens_permited = max(0, min(tool_len, max_new_tokens[selecting_i] - 1 + depth))
                        tool_histories[selecting_i].append({"id": tool_id})

                        end_idx = -depth
                        if end_idx == 0:
                            end_idx = batch_input.shape[1]
                        new_generated_content[selecting_i] = torch.cat((batch_input[selecting_i, initial_prompt_length:end_idx], Tensor(self.tokenized_tools[tool_id]).to(device), OPEN_PARENTHESIS_ID))
                        max_new_tokens[selecting_i] -= tool_len - depth - 1 + 1  # +1 as we will subtract one later. -1 for open parenthesis
                        
                        status[selecting_i] = -1 if max_new_tokens[selecting_i] > 0 else -2


            if self.debug_level > 2:
                LOGGER.debug(f"Sampled: {', '.join([self.decode(sample) for sample in loop_sampled if sample != -1])}")

            # Check if any row wants to use a tool
            # 1, 2 -> 0
            just_sampled_tool = torch.isin(loop_sampled, TOOL_TOKEN_IDS.to(device)).squeeze(1)
            status[just_sampled_tool] = 0
            if self.method_b: max_new_tokens[just_sampled_tool] = torch.min(max_new_tokens[just_sampled_tool], self.max_intention_tokens)


            # Print every row's generated content in the format: 
            # data_i.   loop_i: {loop_i}, status: {decode[generated_content]}, tool_histories: {tool_histories}, current_opts: {current_opts}, 
            #   generated content: {decode[generated_content]}
            if self.debug_level > 1:
                for i in range(batch_indices.shape[0]):
                    data_i = loop_to_data_idx[i].item()
                    LOGGER.debug(f"{data_i}.   loop_i: {i}, status: {status[i].item()}, tool_history: {tool_histories[i]}, current_opts: {current_opts[i]}")
                    # LOGGER.debug(f"Generated content: {self.decode(batch_input[data_i][initial_positions[data_i]:positions[data_i]+ 1])}")
                    if loop_sampled[i] != -1: LOGGER.debug(f"Sampled token: {self.decode(loop_sampled[i])} with id: {loop_sampled[i].item()}")
                    LOGGER.debug("---------------------------------")
                LOGGER.debug("\n\n")

            max_new_tokens -= 1

            # Sequence that reached the stop token
            # 1 -> -1 / -3
            stopped = (status==1) & torch.isin(loop_sampled.squeeze(1), stop_tokens)
            status[stopped] = -1 if arg_selection_mode else -3

            # Rows that reached the max number of tokens, we finish the call
            # any -> -2
            reached_limit = max_new_tokens <= 0
            status[reached_limit] = -2   # Max tokens reached

            finished = (status < 0)
            if finished.any():

                for i in finished.nonzero():
                    # -1: Done
                    # -2: Max tokens  /  - remove generation (free / arg selection mode)
                    # -3: Answer caught
                    data_i = loop_to_data_idx[i].item()

                    if new_generated_content[i] is None:
                        new_generated_content[i] = batch_input[i, initial_prompt_length:].view(-1)

                    if self.method_b:
                        embedding = get_embedding(self.decode(new_generated_content[i]))
                        similarities = [cosine_similarity(tool["embedding"], embedding) for tool in self.tools]
                        # Select tool of maximum score:
                        tool_id = similarities.index(max(similarities))
                        tool_histories[i].append({"id": tool_id})
                        new_generated_content[i] = torch.cat((new_generated_content[i], Tensor(self.tokenized_tools[tool_id]).to(device), OPEN_PARENTHESIS_ID))

                    return_list[data_i] = {
                        "user_prompt": user_prompts[i],
                        "new_content": new_generated_content[i],
                        "tool_history": tool_histories[i],
                        "status": status[i].item(),
                        "tool_use_len": tool_use_len[i],
                    }

                loop_selection_depth = loop_selection_depth[~finished]
                loop_to_data_idx = loop_to_data_idx[~finished]
                batch_indices = batch_indices[:-finished.sum().item()]
                status = status[~finished]
                # tool_histories, batch_generated_count, batch_input, batch_lengths, user_prompts, new_generated_content, loop_sampled, max_new_tokens
                tool_histories = [tool_histories[i] for i in range(len(tool_histories)) if not finished[i]]
                tool_use_len = [tool_use_len[i] for i in range(len(tool_use_len)) if not finished[i]]
                batch_input = batch_input[~finished]
                user_prompts = [user_prompts[i] for i in range(len(user_prompts)) if not finished[i]]
                new_generated_content = [new_generated_content[i] for i in range(len(new_generated_content)) if not finished[i]]
                loop_sampled = loop_sampled[~finished]
                max_new_tokens = max_new_tokens[~finished]
                attention_mask = attention_mask[~finished]
                current_opts = [current_opts[i] for i in range(len(current_opts)) if not finished[i]]
                past_key_values = tuple(tuple(key_value[~finished] for key_value in past_key_value) for past_key_value in past_key_values)

            # Concat the loop sampled content to the batch_input
            batch_input = torch.cat((batch_input, loop_sampled), dim=1)
            attention_mask = torch.cat((attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=torch.long).to(device)), dim=1)


        # Print finish status and info of each:
        ##############WUIGFUIQEWFRGIU"£FRQUWEUILFRWEUCUCRUIQWCUQUIRCU
        for data_id, return_dict in enumerate(return_list):
            if return_dict is not None:
                LOGGER.debug(f"New content: {self.decode(return_dict['new_content'])} \nStatus: {return_dict['status']} - {ROW_STATUS_MEANING[return_dict['status']]}\n")

        return return_list

    


    def forward(self, 
                sentences: List[str],):

        # We receive a batch of texts. 
        LOGGER.info("FORWARD TOOLMASTER")
        LOGGER.info(f"Received batch of {len(sentences)} sentences")

        device = self.device
        self.disable_tools = False

        # We tokenize the texts and store then in tuples with (data_id, tokenized_sentence, generated_content, tool_history)
        pending_completion = [(id, longTensor(self.encode(prompt)).to(device), longTensor([]).to(device), [], 0) for id, prompt in enumerate(sentences)]
        pending_arg_sampling = []
        pending_tool_execution = []

        finished_sentences = {id:{} for id in range(len(sentences))}

        total_finished = 0

        while total_finished < len(sentences):

            #####################################################################################################
            #                        FREE GENERATION MODE AUGMENTED WITH TOOL SELECTION                        #
            #####################################################################################################
            """if sub_indices is None or sub_indices == -1:
            joined_prompts = [torch.cat([prompt, user_prompt]) for prompt, user_prompt in zip(explanation_prompts, user_prompts)]
        elif isinstance(sub_indices, int):
            # Insert user_prompt at sub_indiices
            joined_prompts = [torch.cat([prompt[:sub_indices], user_prompt, prompt[sub_indices:]]) for prompt, user_prompt in zip(explanation_prompts, user_prompts)]
        else:
            joined_prompts = [torch.cat([prompt[:sub_index], user_prompt, prompt[sub_index:]]) for prompt, user_prompt, sub_index in zip(explanation_prompts, user_prompts, sub_indices)]"""


            assert len(pending_completion) + total_finished == len(sentences), f"Lost a sentence in the loop: len completion: {len(pending_completion)}, total_finished: {total_finished}, len sentences: {len(sentences)}"

            LOGGER.info("STARTING FREE GENERATION MODE AUGMENTED WITH TOOL SELECTION")
            batch_i = 0
            batch_size = min(len(pending_completion),self.free_gen_batch_size)
            total_to_complete = len(pending_completion)

            count_finished = 0
            tools_called = [0 for _ in range(len(self.tools))]
            while len(pending_completion) > 0:
                LOGGER.debug(f"Processing batch {batch_i+1}. Sentences processed: {total_to_complete-len(pending_completion)}/{total_to_complete}   ({(total_to_complete - len(pending_completion))/total_to_complete*100:.2f}%))")
                sentence_batch = pending_completion[:batch_size]

                try:
                    ids, user_prompts, generated_content, tool_histories, tool_use_len = zip(*sentence_batch, strict=True)

                    explan_prompt = self.tokenized_free_generation_prompt
                    sub_idx = self.free_gen_sub_idx
                    explanation_prompts = [torch.cat([explan_prompt[:sub_idx], user_prompt, explan_prompt[sub_idx:]]) for user_prompt in user_prompts]

                    output = self.generate(user_prompts = list(generated_content),
                                            explanation_prompts = explanation_prompts,
                                            tool_histories=list(tool_histories),
                                            max_new_tokens = Tensor([self.max_new_tokens - len(gen_content) + use_len for gen_content, use_len in zip(generated_content, tool_use_len, strict=True)]).to(device),
                                            arg_selection_mode = False,
                                            tool_use_len=list(tool_use_len),
                                            temperature=self.temperature,)       
                    
                    assert len(output) == len(ids), f"Output length ({len(output)}) does not match input length ({len(ids)})"
                except torch.cuda.OutOfMemoryError as e: # type: ignore
                    batch_size-=5
                    self.free_gen_batch_size = batch_size
                    LOGGER.info(f"Out of memory error. Reducing batch size to {batch_size}")
                    continue
                
                pending_completion = pending_completion[batch_size:]

                for loop_i, row in enumerate(output):

                    status = row["status"]
                    tool_history = row["tool_history"]
                    id = ids[loop_i]

                    assert status in [-1, -2, -3], "Status should be -1, -2 or -3 for finished sentences."
                    LOGGER.debug(f"Sentence {id} went {status}")

                    gen_content = torch.cat((generated_content[loop_i], row["new_content"]))
                    if status in [-2, -3]:  # Answer caught or max tokens reached
                        response = self.decode(gen_content)
                        finished_sentences[id] = {"user_prompt": sentences[id], "response":response , "tool_history":row["tool_history"], "status":status}
                        count_finished += 1
                    elif status == -1:
                        # print(f"Tool use: {tool_history[-1]}")
                        tools_called[tool_history[-1]["id"]] += 1
                        pending_arg_sampling.append((id, user_prompts[loop_i], gen_content, tool_history,row["tool_use_len"]+2+len(self.tokenized_tools[tool_history[-1]["id"]])))

                batch_i+=1


            LOGGER.info(f"Batch {batch_i+1} processed. Finished sentences: {count_finished}/{total_to_complete}, rest use tools.")
            LOGGER.debug(f"Tools were called the following number of times:")
            for tool_name, tool_count in zip(self.tool_names, tools_called):
                LOGGER.debug(f"{tool_name}: {tool_count}")
            total_finished += count_finished

            LOGGER.debug(torch.cuda.memory_summary(abbreviated=False))
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()

            #####################################################################################################
            #                                   ARGUMENT GENERATION MODE                                        #
            #####################################################################################################
            self.disable_tools = True
            
            batch_size = min(len(pending_arg_sampling),self.arg_sampling_batch_size)
            total_to_generate_args = len(pending_arg_sampling)
            LOGGER.info("STARTING ARGUMENT GENERATION MODE")
            LOGGER.debug(f": {total_to_generate_args}")

            count_finished = 0
            batch_i = 0
            while len(pending_arg_sampling) > 0:
                LOGGER.debug(f"Processing batch {batch_i+1}. Sentences processed: {total_to_generate_args-len(pending_arg_sampling)}/{total_to_generate_args}   ({(total_to_generate_args-len(pending_arg_sampling))/total_to_generate_args*100:.2f}%))")
                
                sentence_batch = pending_arg_sampling[:batch_size]
                try:
                    ids, user_prompts, generated_content, tool_histories, tool_use_len = zip(*sentence_batch, strict=True)

                    explanation_prompts = []
                    max_new_tokens = []
                    for i in range(len(tool_histories)):
                        tool_id = tool_histories[i][-1]["id"]

                        explan_prompt = self.tool_explanation_prompts[tool_id]
                        sub_idx = self.tool_explan_sub_indices[tool_id]
                        explanation_prompts.append(torch.cat([explan_prompt[:sub_idx], user_prompts[i], explan_prompt[sub_idx:]]))
                        max_new_tokens.append(self.max_arg_lengths[tool_id])

                    output = self.generate(user_prompts = list(generated_content),
                                            explanation_prompts = explanation_prompts,
                                            tool_use_len=list(tool_use_len),
                                            tool_histories=list(tool_histories),
                                            max_new_tokens = Tensor(max_new_tokens).to(device),
                                            arg_selection_mode = True,
                                            stop_tokens=self.arg_gen_stoppers,
                                            temperature=self.temperature)
                    
                    assert len(output) == len(ids), f"Output length ({len(output)}) does not match input length ({len(ids)})"

                except torch.cuda.OutOfMemoryError as e: # type: ignore
                    batch_size-=5
                    self.arg_sampling_batch_size = batch_size
                    LOGGER.info(f"Out of memory error. Reducing batch size to {batch_size}")
                    continue

                # REMOVE sampled_args // STRIP )
                pending_arg_sampling = pending_arg_sampling[batch_size:]

                for loop_i, row in renumerate(output):

                    id = ids[loop_i]
                    status = row["status"]
                    row["tool_history"][-1]["args"] = self.decode(row["new_content"])

                    assert status in [-1, -2], "Status should be -1 or -2 for finished sentences in arg gen mode."
                    LOGGER.debug(f"Sentence {id} went {status}")

                    # ARGUMENT GENERATED BABY
                    arg = self.decode(row["new_content"])

                    if len(arg) == 0:
                        # Row is finished
                        response =  self.decode(torch.cat((generated_content[loop_i], row["new_content"], self.close_bad_arg)))
                        finished_sentences[id] = {"user_prompt": sentences[id], "response":response , "tool_history":row["tool_history"], "status":status}
                        count_finished += 1
                        row["tool_history"][-1]["status"] = 2
                         
                        LOGGER.info(f"Model failed to generate arguments for: \ndata: {sentences[id] + response}")
                        LOGGER.info(f"Loop id {loop_i}, data id {id}")
                        LOGGER.info(f"Tool history: {tool_histories[loop_i]}")
                        continue

                    if status == -2:
                        arg = arg.split(")")[0]
                    
                    # Remove last token if it is a ) or an arrow
                    if arg[-1] == "→":
                        arg = arg[:-1]
                    #if arg[-1] == ")":
                    #    arg = arg[:-1]

                    gen_content = torch.cat((generated_content[loop_i], Tensor(self.encode(arg)).long().to(device)))
                    pending_tool_execution.append((id, user_prompts[loop_i], gen_content, tool_histories[loop_i], arg, row["tool_use_len"] + len(row["new_content"])))
                    
                    LOGGER.debug(f"Sentence {id} went -2")
                    tool_histories[loop_i][-1]["args"] = arg

                    """elif status == -2:
                        gen_content = torch.cat((generated_content[loop_i], self.close_bad_arg))
                        tool_histories[loop_i][-1]["status"] = 2

                        if gen_content.shape[0] < self.max_new_tokens:
                            pending_completion.append((id, user_prompts[loop_i], gen_content, tool_histories[loop_i]))
                        else:
                            # Row is finished
                            response =  self.decode(torch.cat((generated_content[loop_i], row["new_content"], self.close_bad_arg)))
                            finished_sentences[id] = {"user_prompt": sentences[id], "response":response , "tool_history":row["tool_history"], "status":status}
                            count_finished += 1
                            
                            LOGGER.info(f"Model failed to generate arguments for: \ndata: {sentences[id] + response}")
                            LOGGER.info(f"Loop id {loop_i}, data id {id}")
                            LOGGER.info(f"Tool history: {tool_histories[loop_i]}")"""

                batch_i+=1

            total_finished += count_finished

            LOGGER.debug(f"Arg sampled with status -1: {len(pending_tool_execution)}")
            LOGGER.debug(f"Arg sampled with status -2 (pending completion): {len(pending_completion)}")
            LOGGER.debug(f"Arg sampled with status -2 (finished as too long): {count_finished}\n")


            LOGGER.debug(torch.cuda.memory_summary(abbreviated=False))

            #####################################################################################################
            #                                          TOOL EXECUTION                                           #
            #####################################################################################################
    
            LOGGER.info("STARTING TOOL EXECUTION")
            count_finished = 0
            if not self.export_tool_execution:
                for i, (id, prompt, generated_content, tool_history, decoded_args, tool_use_len) in enumerate(pending_tool_execution):
                    tool_id = tool_history[-1]["id"]
                    try:
                        parsed_args = self.arg_parsers[tool_id](decoded_args.strip(")"))
                        tool_history[-1]["parsed args"] = parsed_args
                    except Exception as e:
                        LOGGER.warning(f"Error parsing args {decoded_args}")
                        parsed_args = "--Error--"
                        tool_history[-1]["parsed args"] = parsed_args
                        tool_history[-1]["status"] = 3
                        tool_output = f"Parse error: {e}"
                        try:
                            arg = re.match(r"(.*)\)","hdjfh)efkjwef)djf")[0]
                            parsed_args = self.arg_parsers[tool_id](arg)
                            tool_output = self.tools[tool_id](*parsed_args)
                            tool_history[-1]["parsed args"] = parsed_args
                            tool_history[-1]["status"] = 0
                            print("Resurrected tool call")
                        except:
                            pass
                    else:
                        try:
                            tool_output = self.tools[tool_id](*parsed_args)
                            tool_history[-1]["status"] = 0
                        except Exception as e:
                            LOGGER.warning(f"Error executing tool {self.tool_names[tool_id]} with args {parsed_args}")
                            LOGGER.warning(traceback.format_exc())
                            LOGGER.warning(f"Error: {e}")

                            tool_output = e
                            tool_history[-1]["status"] = 3
                        
                        tool_history[-1]["output"] = tool_output

                    LOGGER.debug(f"Executing tool {self.tool_names[tool_id]} with args {decoded_args}, parsed: {parsed_args}, output: {tool_output}")
 
                    tool_output = self.encode("→ " + str(tool_output), truncation=True, max_length=self.max_response_len)
                    tool_output = self.encode(self.decode(tool_output) + END_TOOL_TOKEN, return_tensors="pt")[0].to(device).long()
                    generated_content = torch.cat((generated_content, tool_output))
                    
                    pending_completion.append((id, prompt, generated_content, tool_history, max(tool_use_len + len(tool_output), self.max_exec_tokens)))
                    LOGGER.debug(f"Sentence {id} went -2")

                pending_tool_execution = []

            total_finished += count_finished

            LOGGER.debug(f"Tool execution (pending completion): {len(pending_completion)}")
            LOGGER.debug(f"Tool execution (finished as too long): {count_finished}\n")


        finished_sentences = [finished_sentences[i] for i in range(len(finished_sentences))]

        if self.export_tool_execution:
            return finished_sentences, pending_tool_execution

        return finished_sentences
    