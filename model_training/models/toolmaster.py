import os
import logging
import sys
import traceback

from functools import partial

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import tensor as Tensor

from beartype import beartype
from beartype.typing import List, Callable, Union, Dict, Tuple

# Import parent class of AutoTokenizer
from transformers import LlamaTokenizer, AutoTokenizer, GPT2Tokenizer

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
OPEN_PARENTHESIS = "("
OPEN_PARENTHESIS_ID = 7
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
                     ]

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


# Tools given to the toolmaster must have:
# 1. Name: str - Unique identifier for the tool
# 2. Arg parser: Callable - A function that takes a string and returns a list of arguments
# 3. Tool: Callable - A function that takes a list of argumets and returns a string
# 4. Explanation prompt: Union[torch.Tensor, str] - A string that explains how to use the tool
# 5. Short description: Optional[str] - A short description of the tool



@beartype
class ToolMaster(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        tool_specs: List[Dict],  # of the form {"name": str, "arg_parser": Callable, "tool": Callable, "explanation_prompt": Union[torch.Tensor, str], "short_desc": Optional[str]}
        tokenizer,
        tool_token_ids: List[int],
        end_tool_token = END_TOOL_TOKEN,
        free_generation_prompt: str = None,
        debug_level: int = 0,
        log_dir: str = "/vol/bitbucket/jg2619/augmenting_llms/model_training/models/logs",
        export_tool_execution: bool = False,
        max_new_tokens: int = 30,
        max_response_len: int = 100,
        temperature: float = 0.8,
        catch_answers: bool = False,
        answer_token_ids: List[int] = None,
        post_answer_token_ids: List[int] = None,
        disable_tools:bool = False,
        pretty_tools: bool = False,
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
        print(f"Tool token ids device: {TOOL_TOKEN_IDS.device}")

        OPEN_PARENTHESIS_ID = tokenizer.encode(OPEN_PARENTHESIS)
        assert len(OPEN_PARENTHESIS_ID) == 1, "Open parenthesis token must be a single token"
        OPEN_PARENTHESIS_ID = OPEN_PARENTHESIS_ID[0]

        tool_names = [tool_spec["name"] for tool_spec in tool_specs]
        tokenized_tools = [tokenizer.encode(tool_name) for tool_name in tool_names]
        self.tokenized_tools = tokenized_tools
        self.tool_names = tool_names

        tool_name_desc = []
        for spec in tool_specs:
            name_desc = spec["name"]
            if 'tool_short_desc' in spec:
                name_desc += " (" + spec['short_desc'] +")"
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

        self.close_bad_arg = Tensor(self.encode(f"{PAD_TOKEN})]")[1:]).to(self.device)

        self.tool_explan_sub_indices = []
        for explan in tool_explanation_prompts:
            if "[PROMPT]" in explan:
                self.tool_explan_sub_indices.append(len(self.encode(explan.split("[PROMPT]")[0])))
            else:
                self.tool_explan_sub_indices.append(len(self.encode(explan)))
    
        self.tools = [tool_spec["tool"] for tool_spec in tool_specs]
        self.arg_parsers = [tool_spec["arg_parser"] for tool_spec in tool_specs]
        self.tokenized_tools = tokenized_tools

        self.max_new_tokens = max_new_tokens
        self.max_response_len = max_response_len
        self.temperature = temperature

        print(f"debug_level: {debug_level}")
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

        LOGGER.info("WTF")
        LOGGER.warn("OKAY")
        LOGGER.error("LEL")

        self.catch_answers = catch_answers
        self.answer_token_ids = torch.tensor(answer_token_ids, dtype=torch.int32, device=self.device)
        if post_answer_token_ids is not None:
            post_answer_token_ids = longTensor(post_answer_token_ids).to(self.device)
        self.post_answer_token_ids = post_answer_token_ids

        # Tokens with →
        self.arg_gen_stoppers = []
        for key, value in tokenizer.get_vocab().items():
            if "→" in key or ")]" in key:     # or ")" in key:  # Remove close parenthesis TODO
                self.arg_gen_stoppers.append(value)
        self.arg_gen_stoppers = GPTJ_ARG_STOPPERS
        self.arg_gen_stoppers = Tensor(self.arg_gen_stoppers).to(self.model.device)

        self.export_tool_execution = export_tool_execution

        # COPY PROMPT X TIMES
        # self.tokenized_free_generation_prompt.unsqueeze(0).repeat(batch_size,1)

        for key, value in kwargs.items():
            print(f"Waring: {key} with value {value} is not a valid argument for the model.")

    @torch.no_grad()
    def generate(self, 
                 user_prompts: list[torch.Tensor], 
                 explanation_prompts: Union[List[torch.Tensor], torch.Tensor],
                 generated_content_count: List[int],
                 tool_histories: List[List[Dict]] = None,
                 arg_selection_mode: bool = False,  # Arg selection mode VS free generation augmented with tool selection
                 max_new_tokens: int = 100,
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

        # Position of where to start generating for each row
        positions = Tensor([prompt.shape[0]-1 for prompt in user_prompts]).to(device).unsqueeze(1) + explan_prompt_lens.unsqueeze(1)
        initial_positions = positions.clone() + 1 # Initial positions of each row for data + prompt

        status = Tensor([1 for _ in range(batch_size)]).int().to(device) # Status of each row
        if self.catch_answers and not arg_selection_mode:
            status *= 2
            stop_tokens = self.post_answer_token_ids
        # 0: selecting tool
        # 1: Generating      (Can be stopped by stop tokens)
        # 2: Generating - catching answer (Waiting to see Ans before being able to catch stop tokens)
        # -1: Done
        # -2: Max tokens     (remove gen content if arg selecting)
        # -3: Answer caught

        joined_prompts = [torch.cat((explanation_prompts[i], user_prompts[i])) for i in range(batch_size)] # Joined prompts for each row
        batch_lengths = Tensor([row.shape[0] for row in joined_prompts]).to(device)
        batch_generated_count = Tensor(generated_content_count).to(device)                         # Number of tokens generated for each row
        batch_input = pad_sequence(joined_prompts, padding_value=PAD_ID)
        extra_pad = (batch_lengths + max_new_tokens - batch_generated_count - batch_input.shape[1]).max().item()
        batch_input = F.pad(batch_input, (0, extra_pad,), value=PAD_ID)
        new_generated_content = [longTensor([]).to(device) for _ in range(batch_size)]

        LOGGER.debug(f"Extra pad: {extra_pad}")
        LOGGER.debug(f"Batch lengths: {batch_lengths}")
        LOGGER.debug(f"Max new tokens: {max_new_tokens}")
        LOGGER.debug(f"Generated content lens: {generated_content_count}")
        LOGGER.debug(f"Batch input shape: {batch_input.shape}")
        LOGGER.debug("Primes:")
        for i, row in enumerate(batch_input):
            LOGGER.debug(self.decode(row))
            LOGGER.debug(f"Position: {positions[i].item()}, token_id: {row[positions[i].item()].item()}, token: {self.decode(row[positions[i].item()].item())}")
            LOGGER.debug(f"Position + 1: {positions[i].item()+1}, token_id: {row[positions[i].item()+1].item()}, token: {self.decode(row[positions[i].item()+1].item())}")

        LOGGER.debug(f"Stop tokens: {stop_tokens}")
        
        # Indexing tensor utils
        loop_to_data_idx = torch.arange(batch_size).to(device)                 # Mapping from loop index to batch index
        batch_indices = torch.arange(batch_size).to(device).unsqueeze(1)       # ARANGE THAT ADJUSTS TO THE LOOP BATCH SIZE AS SAMPLES FINISH
        
        # Tool selection utils
        loop_selection_depth = torch.zeros(batch_size).int().to(device)        # Depth of the tool selection tree
        current_opts = [self.tool_selection_dict for _ in range(batch_size)]   # Current tool options for row
        tools_are_disabled = self.disable_tools or arg_selection_mode


        return_list = [None for _ in range(batch_size)]                        # List of return values for each row
        
        while batch_indices.shape[0] > 0:                                                                

            # Remove assertion: TODO
            assert loop_to_data_idx.shape[0] == batch_indices.shape[0], "Loop to data index and batch indices must have the same shape"

            # MODEL FORWARD CALL. MAINTAINS SHAPE EVEN AFTER INDEXING
            #print(f"Input shape {batch_input.shape}")
            #print(f"Positions shape {positions.shape}")
            #print(f"Loop to data idx shape {loop_to_data_idx.shape}")
            #print(f"Batch indices shape {batch_indices.shape}")
            #print(positions)
            input = self.model.prepare_inputs_for_generation(batch_input[loop_to_data_idx], past=past_key_values)
            output = self.model(input)

            loop_last_logits = output.logits[batch_indices, positions[loop_to_data_idx] + LOGIT_DISPLACEMENT]
            #loop_last_logits[:, :, TOOL_TOKEN_IDS[0]] += 10   #TOOL_TOKEN_ID , 13 fulstp

            positions[loop_to_data_idx] += 1

            if tools_are_disabled:   # Tool usage not available
                loop_last_logits[:, :, TOOL_TOKEN_IDS] = -1e10

            # Gumbel sample for rows not selecting a tool. Tool selection has different sampling procedure
            sample_ids = loop_to_data_idx[status>0]
            loop_sampled = torch.ones(batch_indices.shape[0], 1).long().to(device)*-1
            loop_sampled[status>0] = gumbel_sample(loop_last_logits[status>0], temperature=temperature)
            batch_input[sample_ids.unsqueeze(1), positions[sample_ids]] = loop_sampled[status>0]

            if self.debug_level > 2:
                LOGGER.debug(f"Sampled: {', '.join([self.decode(sample) for sample in loop_sampled if sample != -1])}")

            # Catch answers     2 -> 1
            if not arg_selection_mode and self.catch_answers:
                # Check if any of the tokens are answer tokens
                caught_answers = torch.isin(loop_sampled, self.answer_token_ids).squeeze(1).bool()
                status[caught_answers] = 1   # Generating status (can be stopped)

            # Sampling procedure for rows selecting a tool
            # 0 -> -1, -2
            for selecting_i in (status==0).nonzero().squeeze(1):
                data_i = loop_to_data_idx[selecting_i].item()
                    # Tool names are composed of tokens. ie. [CAL] [CUL] [ATOR]. We call each token a syllable
                    # Options for the next syllable. 
                syllable_opts = Tensor(list(current_opts[data_i].keys())).to(device)
                next_syllable_idx = loop_last_logits[selecting_i,0,syllable_opts].argmax(dim=-1)
                next_syllable = syllable_opts[next_syllable_idx].item() 
                batch_input[data_i, positions[data_i]] = next_syllable
                loop_selection_depth[selecting_i] += 1
                current_opts[data_i] = current_opts[data_i][next_syllable]

                # If current opts is a dict, there is a tie between possible tools. We need to keep selecting syllables.
                # ELSE: We've reached a tool id
                if not isinstance(current_opts[data_i], dict):
                    tool_id = current_opts[data_i]
                    depth = loop_selection_depth[selecting_i].item()-1      # Selection_depth = i means we've selected the ith syllable of tool name. -1 for indexing purposes.
                    tool_len = len(self.tokenized_tools[tool_id])

                    tokens_permited = max(0, min(tool_len, max_new_tokens - batch_generated_count[data_i] - 1 + depth))
                    tool_histories[data_i].append({"id": tool_id})

                    # print(f"Took {tokens_permited} tokens from tool {tool_id}: {self.decode(self.tokenized_tools[tool_id])} and len {tool_len}")
                    batch_input[data_i, positions[data_i]-depth:positions[data_i]-depth+tokens_permited] = Tensor(self.tokenized_tools[tool_id][:tokens_permited]).to(device)
                    if tokens_permited > 0: 
                        batch_input[data_i, positions[data_i]-depth+tokens_permited] = OPEN_PARENTHESIS_ID
                        status[selecting_i] = -1
                    else:
                        status[selecting_i] = -2
                        
                    batch_generated_count[data_i] += tool_len - depth  # +1 for open parenthesis will be added in the standsrd +1 increment laters
                    positions[data_i] += tokens_permited               # Leave cursor at last generated position to extract generated content


            # Check if any row wants to use a tool
            # 1 / 2 -> 0
            just_sampled_tool = torch.isin(loop_sampled, TOOL_TOKEN_IDS.to(device)).squeeze(1)
            status[just_sampled_tool] = 0


            # Print every row's generated content in the format: 
            # data_i.   loop_i: {loop_i}, status: {decode[generated_content]}, tool_histories: {tool_histories}, current_opts: {current_opts}, 
            #   generated content: {decode[generated_content]}
            if self.debug_level > 1:
                for i in range(batch_indices.shape[0]):
                    data_i = loop_to_data_idx[i].item()
                    LOGGER.debug(f"{data_i}.   loop_i: {i}, status: {status[i].item()}, tool_history: {tool_histories[data_i]}, current_opts: {current_opts[data_i]}")
                    LOGGER.debug(f"Generated content: {self.decode(batch_input[data_i][initial_positions[data_i]:positions[data_i]+ 1])}")
                    if loop_sampled[i] != -1: LOGGER.debug(f"Sampled token: {self.decode(loop_sampled[i])} with id: {loop_sampled[i].item()}")
                    LOGGER.debug("---------------------------------")
                LOGGER.debug("\n\n")
            batch_generated_count[loop_to_data_idx] += 1

            # Sequence that reached the stop token
            # 1 -> -1 / -3
            stopped = (status==1) & torch.isin(loop_sampled.squeeze(1), stop_tokens)
            status[stopped] = -1 if arg_selection_mode else -3

            # Rows that reached the max number of tokens, we finish the call
            # any -> -2
            reached_limit = batch_generated_count[loop_to_data_idx] >= max_new_tokens
            status[reached_limit] = -2   # Max tokens reached

            finished = (status < 0)
            if finished.any():

                for i in finished.nonzero():
                    # -1: Done
                    # -2: Max tokens  /  - remove generation (free / arg selection mode)
                    # -3: Answer caught
                    data_i = loop_to_data_idx[i].item()

                    new_generated_content[data_i] = batch_input[data_i, initial_positions[data_i]:positions[data_i]+1]

                    return_list[data_i] = {
                        "user_prompt": user_prompts[data_i],
                        "new_content": new_generated_content[data_i],
                        "tool_history": tool_histories[data_i],
                        "status": status[i].item(),
                    }

                loop_selection_depth = loop_selection_depth[~finished]
                loop_to_data_idx = loop_to_data_idx[~finished]
                batch_indices = batch_indices[:-finished.sum().item()]
                status = status[~finished]

        # Print finish status and info of each:
        for data_i, return_dict in enumerate(return_list):
            if return_dict is not None:
                LOGGER.debug(f"New content: {self.decode(return_dict['new_content'])} \nStatus: {return_dict['status']} - {ROW_STATUS_MEANING[return_dict['status']]}\n")

        return return_list

    


    def forward(self, 
                sentences: List[str],):

        # We receive a batch of texts. 
        LOGGER.info("FORWARD TOOLMASTER")
        LOGGER.info(f"Received batch of {len(sentences)} sentences")

        device = self.device

        # We tokenize the texts and store then in tuples with (data_id, tokenized_sentence, generated_content, tool_history)
        pending_completion = [(id, longTensor(self.encode(prompt)).to(device), longTensor([]).to(device), []) for id, prompt in enumerate(sentences)]
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
            batch_size = 11
            total_to_complete = len(pending_completion)

            count_finished = 0
            tools_called = [0 for _ in range(len(self.tools))]
            while len(pending_completion) > 0:
                LOGGER.debug(f"Processing batch {batch_i+1}. Sentences processed: {total_to_complete-len(pending_completion)}/{total_to_complete}   ({(total_to_complete - len(pending_completion))/total_to_complete*100:.2f}%))")
                sentence_batch = pending_completion[:batch_size]

                try:
                    ids, user_prompts, generated_content, tool_histories = zip(*sentence_batch, strict=True)

                    explan_prompt = self.tokenized_free_generation_prompt
                    sub_idx = self.free_gen_sub_idx
                    explanation_prompts = [torch.cat([explan_prompt[:sub_idx], user_prompt, explan_prompt[sub_idx:]]) for user_prompt in user_prompts]

                    output = self.generate(user_prompts = list(generated_content),
                                            explanation_prompts = explanation_prompts,
                                            generated_content_count=[len(gen_content) for gen_content in generated_content],
                                            tool_histories=list(tool_histories),
                                            max_new_tokens = self.max_new_tokens,
                                            arg_selection_mode = False,
                                            temperature=self.temperature,)       
                    
                    assert len(output) == len(ids), f"Output length ({len(output)}) does not match input length ({len(ids)})"
                except torch.cuda.OutOfMemoryError as e: # type: ignore
                    batch_size-=5
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
                    if status == -2 and len(tool_history) > 0:
                        tool_history[-1]["status"] = 1
                    elif status == -1:
                        # print(f"Tool use: {tool_history[-1]}")
                        tools_called[tool_history[-1]["id"]] += 1
                        pending_arg_sampling.append((id, user_prompts[loop_i], gen_content, tool_history,))

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

            batch_size = 11
            total_to_generate_args = len(pending_arg_sampling)
            LOGGER.info("STARTING ARGUMENT GENERATION MODE")
            LOGGER.debug(f": {total_to_generate_args}")

            count_finished = 0
            batch_i = 0
            while len(pending_arg_sampling) > 0:
                LOGGER.debug(f"Processing batch {batch_i+1}. Sentences processed: {total_to_generate_args-len(pending_arg_sampling)}/{total_to_generate_args}   ({(total_to_generate_args-len(pending_arg_sampling))/total_to_generate_args*100:.2f}%))")
                
                sentence_batch = pending_arg_sampling[:batch_size]
                try:
                    ids, user_prompts, generated_content, tool_histories = zip(*sentence_batch, strict=True)

                    explanation_prompts = []
                    for i, hist in enumerate(tool_histories):
                        tool_id = hist[-1]["id"]

                        explan_prompt = self.tokenized_free_generation_prompt
                        sub_idx = self.tool_explan_sub_indices[tool_id]
                        explanation_prompts.append(torch.cat([explan_prompt[:sub_idx], user_prompts[i], explan_prompt[sub_idx:]]))

                    output = self.generate(user_prompts = list(generated_content),
                                            explanation_prompts = explanation_prompts,
                                            generated_content_count=[len(gen_content) for gen_content in generated_content],
                                            tool_histories=list(tool_histories),
                                            max_new_tokens = self.max_new_tokens,
                                            arg_selection_mode = True,
                                            stop_tokens=self.arg_gen_stoppers,
                                            temperature=self.temperature)
                    
                    assert len(output) == len(ids), f"Output length ({len(output)}) does not match input length ({len(ids)})"

                except torch.cuda.OutOfMemoryError as e: # type: ignore
                    batch_size-=5
                    LOGGER.info(f"Out of memory error. Reducing batch size to {batch_size}")
                    continue

                # REMOVE sampled_args // STRIP )
                pending_arg_sampling = pending_arg_sampling[batch_size:]

                for loop_i, row in renumerate(output):

                    id = ids[loop_i]
                    status = row["status"]
                    tool_histories[loop_i][-1]["args"] = self.decode(row["new_content"])

                    assert status in [-1, -2], "Status should be -1 or -2 for finished sentences in arg gen mode."
                    LOGGER.debug(f"Sentence {id} went {status}")

                    if status == -1:
                        # ARGUMENT GENERATED BABY
                        gen_content = torch.cat((generated_content[loop_i], row["new_content"]))
                        pending_tool_execution.append((id, user_prompts[loop_i], gen_content, tool_histories[loop_i], row["new_content"][:-1]))
                        tool_histories[loop_i][-1]["args"] = self.decode(row["new_content"])

                    elif status == -2:
                        gen_content = torch.cat((generated_content[loop_i], self.close_bad_arg))
                        tool_histories[loop_i][-1]["status"] = 2
                        if len(gen_content) >= self.max_new_tokens:
                            # Row is finished
                            response =  self.decode(torch.cat((generated_content[loop_i], row["new_content"], self.close_bad_arg)))
                            finished_sentences[id] = {"user_prompt": sentences[id], "response":response , "tool_history":row["tool_history"], "status":status}
                            count_finished += 1
                            
                            LOGGER.warning(f"Model failed to generate arguments for: \ndata: {sentences[id] + response}")
                            LOGGER.warning(f"Loop id {loop_i}, data id {id}")
                            LOGGER.warning(f"Tool history: {tool_histories[loop_i]}")
                        else:
                            pending_completion.append((id, user_prompts[loop_i], gen_content, tool_histories[loop_i]))

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
                for i, (id, prompt, generated_content, tool_history, sampled_args) in enumerate(pending_tool_execution):
                    tool_id = tool_history[-1]["id"]
                    try:
                        parsed_args = self.arg_parsers[tool_id](self.decode(sampled_args).strip(")"))
                        tool_history[-1]["parsed args"] = parsed_args
                    except Exception as e:
                        LOGGER.warning(f"Error parsing args {self.decode(sampled_args)}")
                        parsed_args = "--Error--"
                        tool_history[-1]["parsed args"] = parsed_args
                        tool_history[-1]["status"] = 3
                        tool_output = f"Parse error: {e}"
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

                    LOGGER.debug(f"Executing tool {self.tool_names[tool_id]} with args {self.decode(sampled_args)}, parsed: {parsed_args}, output: {tool_output}")
 
                    tool_output = self.encode(")→ " + str(tool_output), truncation=True, max_length=self.max_response_len)
                    tool_output = self.encode(self.decode(tool_output) + END_TOOL_TOKEN, return_tensors="pt")[0].to(device).long()
                    generated_content = torch.cat((generated_content[:-1], tool_output))
                    
                    if generated_content.shape[0] < self.max_new_tokens:
                        pending_completion.append((id, prompt, generated_content, tool_history))
                        LOGGER.debug(f"Sentence {id} went -2")
                    else:
                        LOGGER.debug(f"Sentence {id} went -1")
                        finished_sentences[id] = {"user_prompt":self.decode(prompt), "response":self.decode(generated_content), "tool_history":tool_history, "status":-2, "batch_id":i}
                        count_finished += 1

                pending_tool_execution = []

            total_finished += count_finished

            LOGGER.debug(f"Tool execution (pending completion): {len(pending_completion)}")
            LOGGER.debug(f"Tool execution (finished as too long): {count_finished}\n")


        finished_sentences = [finished_sentences[i] for i in range(len(finished_sentences))]

        if self.export_tool_execution:
            return finished_sentences, pending_tool_execution

        return finished_sentences
    