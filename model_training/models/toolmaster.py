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
logging.shutdown()
reload(logging)

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
        free_generation_prompt: str = None,
        debug_level: int = 0,
        log_dir: str = "/vol/bitbucket/jg2619/augmenting_llms/model_training/models/logs",
        export_tool_execution: bool = False,
        max_new_tokens: int = 30,
        max_response_len: int = 100,
        temperature: float = 0.8,
    ): 
        super().__init__()

        global PAD_ID, PAD_TOKEN, OPEN_PARENTHESIS_ID, TOOL_TOKEN_IDS
        
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

        if free_generation_prompt is None:
            free_generation_prompt = "You can use these tools to help you answer: " + ", ".join(tool_name_desc) + ".\n\n"
        
        self.free_generation_prompt = free_generation_prompt
        self.tokenized_free_generation_prompt = longTensor(tokenizer.encode(self.free_generation_prompt)).to(self.device)

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


        self.tool_selection_dict = tool_selection_dict
        tool_explanation_prompts = [tool_spec["explanation_prompt"] for tool_spec in tool_specs]
        if isinstance(tool_explanation_prompts[0], str):
            tool_explanation_prompts = [longTensor(tokenizer.encode(prompt)) for prompt in tool_explanation_prompts]
        self.tool_explanation_prompts = [prompt.to(self.device) for prompt in tool_explanation_prompts]
        self.tools = [tool_spec["tool"] for tool_spec in tool_specs]
        self.arg_parsers = [tool_spec["arg_parser"] for tool_spec in tool_specs]
        self.tokenized_tools = tokenized_tools

        self.max_new_tokens = max_new_tokens
        self.max_response_len = max_response_len
        self.temperature = temperature

        self.debug_level = debug_level
        # Create log dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # count files in log dir
        i = len(os.listdir(log_dir))
        logging.basicConfig(filename=f'{log_dir}/{i}.log', level=logging.DEBUG if debug_level>0 else logging.INFO, format='%(asctime)s:  %(message)s', datefmt='%m/%d/%Y %I:%M:%S  ')
        print(f"Logging to {log_dir}/{i}.log")
        

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:  %(message)s')
        handler.setFormatter(formatter)

        logger_root = logging.getLogger()
        logger_root.addHandler(handler)

        # Tokens with →
        self.arg_gen_stoppers = []
        for key, value in tokenizer.get_vocab().items():
            if "→" in key or ")" in key:  # Remove close parenthesis TODO
                self.arg_gen_stoppers.append(value)
        self.arg_gen_stoppers = Tensor(self.arg_gen_stoppers).to(self.model.device)

        self.export_tool_execution = export_tool_execution

        # COPY PROMPT X TIMES
        # self.tokenized_free_generation_prompt.unsqueeze(0).repeat(batch_size,1)

    @torch.no_grad()
    def generate(self, 
                 primes: list[torch.Tensor], 
                 prompts: Union[List[torch.Tensor], torch.Tensor],
                 generated_content: List[torch.Tensor] = None,
                 tool_history: List[List[Dict]] = None,
                 arg_selection_mode: bool = False,  # Arg selection mode VS free generation augmented with tool selection
                 max_new_tokens: int = 100,
                 temperature: float = 0.8, 
                 stop_tokens: Union[List[int],int,torch.Tensor] = [],):
        
        global PAD_ID, PAD_TOKEN, OPEN_PARENTHESIS_ID, TOOL_TOKEN_IDS, LOGIT_DISPLACEMENT

        device = self.device

        # Each data point as it goes through the loop should have:
        # 1. An updated prime: The data point to be completed including the generated content
        # 2. The original prime: The prompt that was prepended to the prime
        # 3. Total generated content: The generated content for the data point
        # 4. Current generated content: The last generated content for the data point in the current loop section
        # 5. The tool history



        if isinstance(prompts, torch.Tensor):
            prompts = [prompts for _ in range(len(primes))]

        assert primes[0].dim() == 1, "Primes must be 1D tensor with the tokenized data"
        assert prompts[0].dim() == 1, "Prompt must be 1D tensor with the tokenized prompt"

        # Device assertions
        assert primes[0].device == device, "Primes must be on the same device as the model"
        assert prompts[0].device == device, "Prompts must be on the same device as the model"
        if generated_content is not None:
            assert generated_content[0].device == device, "Generated content must be on the same device as the model"
        
        batch_size = len(primes)                                                  # BATCH SIZE
        prompt_lens = Tensor([prompt.shape[0] for prompt in prompts]).to(device)  # LENGTHS OF PREPENDED PROMPTS
        
        if tool_history is None:        # History of tools used for each row
            tool_history = [[] for _ in range(batch_size)]
        if generated_content is None:   # Generated content for each row
            generated_content = [longTensor([]).to(device) for _ in range(batch_size)]
        generated_content_lens = Tensor([content.shape[0] for content in generated_content]).to(device) 
        new_generated_content = [longTensor([]).to(device) for _ in range(batch_size)]
        if isinstance(stop_tokens, int):
            stop_tokens = [stop_tokens]
        if not isinstance(stop_tokens, torch.Tensor):
            stop_tokens = longTensor(stop_tokens).to(device).view(-1)

        # Position of where to start generating for each row
        positions = Tensor([prime.shape[0]-1 for prime in primes]).to(device).unsqueeze(1)
        positions += prompt_lens.unsqueeze(1) + generated_content_lens.unsqueeze(1) # Add prompt and generated content lengths
        initial_positions = positions.clone() # Initial positions of each row for data + prompt

        batch_input = [torch.cat([prompt, prime, content]) for prompt, prime, content in zip(prompts, primes, generated_content)]
        batch_lengths = Tensor([row.shape[0] for row in batch_input]).to(device)
        batch_input = pad_sequence(batch_input, padding_value=PAD_ID)
        extra_pad = (batch_lengths + max_new_tokens - generated_content_lens - batch_input.shape[1]).max().item()
        batch_input = F.pad(batch_input, (0, extra_pad,), value=PAD_ID)

        print(f"Extra pad: {extra_pad}")
        print(f"Batch lengths: {batch_lengths}")
        print(f"Max new tokens: {max_new_tokens}")
        print(f"Generated content lens: {generated_content_lens}")
        print(f"Batch input shape: {batch_input.shape}")
        
        # Indexing tensor utils
        loop_to_data_idx = torch.arange(batch_size).to(device)                 # Mapping from loop index to batch index
        batch_indices = torch.arange(batch_size).to(device).unsqueeze(1)       # ARANGE THAT ADJUSTS TO THE LOOP BATCH SIZE AS SAMPLES FINISH
        
        # Tool selection utils
        loop_selection_depth = torch.zeros(batch_size).int().to(device)        # Depth of the tool selection tree
        loop_is_selecting_tools = torch.zeros(batch_size).bool().to(device)    # Indices where we are selecting a tool
        current_opts = [self.tool_selection_dict for _ in range(batch_size)]               # Current tool options for row

        batch_generated_count = generated_content_lens.clone() # Number of tokens generated for each row

        while batch_generated_count.min().item() < max_new_tokens and batch_indices.shape[0] > 0:

            # Remove assertion:
            assert loop_to_data_idx.shape[0] == batch_indices.shape[0], "Loop to data index and batch indices must have the same shape"
            # Shapes:
            # b = batch_size
            # l = loop_batch_size
            # s = padded_prompt_size

            # batch_input: (b, s)
            # batch_generated_count: (b)
            # batch_indices: (l, 1)
            # loop_to_data_idx: (l)
            # positions: (l, 1)
            # loop_is_selecting_tools: (l)
            # loop_selection_depth: (l)
            # initial_positions: (b, 1)

            # MODEL FORWARD CALL. MAINTAINS SHAPE EVEN AFTER INDEXING
            print(f"Input shape {batch_input.shape}")
            print(f"Positions shape {positions.shape}")
            print(f"Loop to data idx shape {loop_to_data_idx.shape}")
            print(f"Batch indices shape {batch_indices.shape}")
            print(positions)
            output =self.model(batch_input[loop_to_data_idx], use_cache=False)
            loop_last_logits = output.logits[batch_indices, positions[loop_to_data_idx] + LOGIT_DISPLACEMENT]
            #loop_last_logits[:, :, TOOL_TOKEN_IDS[0]] += 10   #TOOL_TOKEN_ID , 13 fulstp

            positions[loop_to_data_idx] += 1


            if arg_selection_mode:   # Tool usage not available
                loop_last_logits[:, :, TOOL_TOKEN_IDS] = -1e10

            # Gumbel sample for rows not selecting a tool. Tool selection has different sampling procedure
            sample_ids = loop_to_data_idx[~loop_is_selecting_tools]
            loop_sampled = torch.ones(batch_indices.shape[0], 1).long().to(device)*-1
            loop_sampled[~loop_is_selecting_tools] = gumbel_sample(loop_last_logits[~loop_is_selecting_tools], temperature=temperature)
            print(loop_sampled)
            batch_input[sample_ids.unsqueeze(1), positions[sample_ids]] = loop_sampled[~loop_is_selecting_tools]
            batch_generated_count[sample_ids] += 1

            # Sampling procedure for rows selecting a tool
            if loop_is_selecting_tools.any():
                print("SELECTING TOOLS LOOP")
                for selecting_i in reversed(loop_is_selecting_tools.nonzero().squeeze(1)):
                    
                    data_i = loop_to_data_idx[selecting_i].item()
                        # Tool names are composed of tokens. ie. [CAL] [CUL] [ATOR]. We call each token a syllable
                        # Options for the next syllable. 
                    syllable_opts = Tensor(list(current_opts[data_i].keys())).to(device)
                    next_syllable_idx = loop_last_logits[selecting_i,0,syllable_opts].argmax(dim=-1)
                    next_syllable = syllable_opts[next_syllable_idx].item()
                    batch_input[data_i, positions[data_i]] = next_syllable
                    loop_selection_depth[selecting_i] += 1
                    current_opts[data_i] = current_opts[data_i][next_syllable]

                    batch_generated_count[data_i] += 1

                    # If current opts is a dict, there is a tie between possible tools. We need to keep selecting syllables.
                    if not isinstance(current_opts[data_i], dict):   # ELSE: We've reached a tool id
                        tool_id = current_opts[data_i]
                        depth = loop_selection_depth[selecting_i].item()-1   # Selection_depth = i means we've selected the ith syllable of tool name. -1 for indexing purposes.
                        tool_len = len(self.tokenized_tools[tool_id])

                        batch_generated_count[data_i] += tool_len + 1 - depth - 1 # +1 for open parenthesis

                        if batch_generated_count[data_i] >= max_new_tokens:
                            logging.warning(f"Stopping generation at row {data_i} that reached the generation limit")
                            logging.warning(f"Data: {self.decode(batch_input[data_i])}")
                            logging.warning(f"Data id {data_i}")
                            logging.warning(f"Tool history: {tool_history[data_i]}")
                            positions[data_i] = -1
                        else:
                            batch_input[data_i, positions[data_i]-depth:positions[data_i]-depth+tool_len] = Tensor(self.tokenized_tools[tool_id]).to(device)
                            batch_input[data_i, positions[data_i]-depth+tool_len] = OPEN_PARENTHESIS_ID

                            new_generated_content[data_i] = batch_input[data_i, initial_positions[data_i]+1:positions[data_i]-depth+tool_len+1]

                            tool_history[data_i].append({"id": tool_id})

                        # Remove index i
                        remove_index = torch.arange(loop_to_data_idx.shape[0]).to(device) != selecting_i
                        loop_is_selecting_tools = loop_is_selecting_tools[remove_index]
                        loop_selection_depth = loop_selection_depth[remove_index]
                        loop_to_data_idx = loop_to_data_idx[remove_index]
                        loop_sampled = loop_sampled[remove_index]
                        batch_indices = batch_indices[:-1]

                    

            print(f"Sampled: {', '.join([self.decode(sample) for sample in loop_sampled if sample != -1])}")
            print("Primes:")
            for row in batch_input:
                print(self.decode(row))

            # Check if any row wants to use a tool
            print(f"Device of loop_sampled: {loop_sampled.device}")
            print(f"Device of tool token ids: {TOOL_TOKEN_IDS.device}")
            just_sampled_tool = torch.isin(loop_sampled, TOOL_TOKEN_IDS.to(device)).view(-1)
            if (just_sampled_tool).any():   # New rows selecting tools!
                print("JUST SELECTED TOOL")
                print(loop_is_selecting_tools)
                print(just_sampled_tool)
                loop_is_selecting_tools[just_sampled_tool] = True

            # Rows that reached the max number of tokens, we finish the call
            reached_limit = batch_generated_count[loop_to_data_idx] >= max_new_tokens
            # Sequence that reached the stop token
            finished = torch.isin(loop_sampled.squeeze(1), stop_tokens) + reached_limit
            if finished.any():
                print(f"{finished.sum()} FINISHED")
                print(f"Reached limit: {reached_limit.sum()}")
                print(f"Finished tensor: {finished}")
                print(f"reached limit tensor: {reached_limit}")

                for finished_i in finished.nonzero().squeeze(1):
                    data_i = loop_to_data_idx[finished_i].item()
                    new_generated_content[data_i] = batch_input[data_i, initial_positions[data_i]+1:positions[data_i]+1]
                    if reached_limit[finished_i]:
                        logging.warning(f"Stopping generation at row {data_i} that reached the generation limit")
                        logging.warning(f"Data: {self.decode(batch_input[data_i])}")
                        if arg_selection_mode:
                            # model failed to generate arguments.
                            logging.warning(f"Model failed to generate arguments for: \ndata: {self.decode(batch_input[data_i])}")
                            logging.warning(f"Data id {data_i}")
                            logging.warning(f"Tool history: {tool_history[data_i]}")
                            tool_history[data_i][-1]["status"] = "Failed to generate arguments"
                            positions[data_i] = -1    # This marks tool use error - rectifies use and resumes generation
                            new_generated_content[data_i] = longTensor([]).to(device)


                if not arg_selection_mode:
                    # These rows are done generating. Mark them as finished
                    positions[loop_to_data_idx[finished]] = -1

                loop_to_data_idx = loop_to_data_idx[~finished]
                loop_is_selecting_tools = loop_is_selecting_tools[~finished]
                loop_selection_depth = loop_selection_depth[~finished]
                batch_indices = batch_indices[:-finished.sum().item()]


        output = {
            "primes": primes,
            "generated_content": [torch.cat([content, new_content]) for content, new_content in zip(generated_content, new_generated_content)],
            "tool_history": tool_history,
            "status": positions,
            "sampled_args": [arg[:-1] for arg in new_generated_content],
        }
        if not arg_selection_mode:
            del output["sampled_args"]

        return output

    


    def forward(self, 
                sentences: List[str],):

        # We receive a batch of texts. 
        logging.info("FORWARD TOOLMASTER")
        logging.info(f"Received batch of {len(sentences)} sentences")

        device = self.device

        # We tokenize the texts and store then in tuples with (tokenized_sentence, pos, count generation, tool_history)
        pending_completion = [(longTensor(self.encode(prime)).to(device), longTensor([]).to(device), []) for prime in sentences]
        finished_sentences = []
        ids = [i for i in range(len(sentences))]

        while len(pending_completion) > 0:

            ####################################################
            # FREE GENERATION MODE AUGMENTED WITH TOOL SELECTION
            ####################################################
            logging.info("STARTING FREE GENERATION MODE AUGMENTED WITH TOOL SELECTION")
            i = 0
            batch_size = 11
            pending_arg_sampling = []
            pending_count = len(pending_completion)

            while pending_count > 0:
                logging.debug(f"Processing batch {i+1}. Sentences processed: {len(pending_completion)-pending_count}/{len(pending_completion)}   ({(len(pending_completion)-pending_count)/len(pending_completion)*100:.2f}%))")
                start_idx = max(0, pending_count-batch_size)
                sentence_batch = pending_completion[start_idx:pending_count]

                try:
                    primes, generated_content, tool_history = zip(*sentence_batch)
                    output_dict = self.generate(primes = [prime for prime in primes],
                                                generated_content=list(generated_content),
                                                prompts = self.tokenized_free_generation_prompt,
                                                tool_history=list(tool_history),
                                                max_new_tokens = self.max_new_tokens,
                                                arg_selection_mode = False,
                                                temperature=self.temperature,)            
                except torch.cuda.OutOfMemoryError as e: # type: ignore
                    batch_size-=5
                    sentence_batch = sentence_batch[5:]
                    logging.info(f"Out of memory error. Reducing batch size to {batch_size}")
                    continue
                
                pending_count -= len(sentence_batch)
                finished_count = 0
                tools_called = [0 for _ in range(len(self.tools))]
                for i, (prime, generated_content, tool_history, status) in renumerate(list(zip(*output_dict.values()))):
                    if status == -1:
                        finished_sentences.append({"id":ids.pop(i), "original_prime": self.decode(prime.cpu()), "response":self.decode(generated_content.cpu()), "tool history":tool_history})
                        finished_count += 1
                    else:
                        print(f"Tool use: {tool_history[-1]}")
                        tools_called[tool_history[-1]["id"]] += 1
                        pending_arg_sampling.append((prime, generated_content, tool_history,))

                logging.info(f"Batch {i+1} processed. Finished sentences: {finished_count}/{len(sentence_batch)}, rest use tools.")
                logging.info(f"Tools were called the following number of times:")
                for tool_name, tool_count in zip(self.tool_names, tools_called):
                    logging.info(f"{tool_name}: {tool_count}")
                i+=1


            ####################################################
            # ARGUMENT GENERATION MODE
            ####################################################

            logging.info("STARTING ARGUMENT GENERATION MODE")
            batch_size = 11
            pending_completion = []
            pending_tool_execution = []
            total_pending_args = len(pending_arg_sampling)
            pending_count = total_pending_args
            logging.info(f"Pending: {pending_arg_sampling}")

            while pending_count > 0:
                logging.debug(f"Processing batch {i+1}. Sentences processed: {len(pending_arg_sampling)-pending_count}/{len(pending_arg_sampling)}   ({(len(pending_arg_sampling)-pending_count)/len(pending_arg_sampling)*100:.2f}%))")
                
                print(pending_arg_sampling)
                start_idx = max(0, pending_count-batch_size)
                sentence_batch = pending_arg_sampling[start_idx:pending_count]
                try:
                    prime, generated_content, tool_histories = zip(*sentence_batch)
                    prompts = [self.tool_explanation_prompts[hist[-1]["id"]] for hist in tool_histories]

                    print("ARG GEN PROMPTS")
                    print(prompts)
                    output_dict = self.generate(primes = list(prime),
                                                generated_content=list(generated_content),
                                                prompts = prompts,
                                                tool_history=list(tool_histories),
                                                max_new_tokens = self.max_new_tokens,
                                                arg_selection_mode = True,
                                                stop_tokens=self.arg_gen_stoppers,
                                                temperature=self.temperature,)
                except torch.cuda.OutOfMemoryError as e: # type: ignore
                    batch_size-=5
                    sentence_batch = sentence_batch[5:]
                    logging.info(f"Out of memory error. Reducing batch size to {batch_size}")
                    continue
                
                pending_count -= len(sentence_batch)
                finished_count = 0
                for i, (prime, generated_content, tool_history, status, sampled_args) in enumerate(zip(*output_dict.values())):
                    if status == -1:
                        pending_completion.append((prime, generated_content, tool_history))
                        finished_count += 1
                    else:
                        # TOOL SELECTION BABY
                        pending_tool_execution.append((prime, generated_content, tool_history, sampled_args))

            ####################################################
            # TOOL EXECUTION
            ####################################################

            logging.info("STARTING TOOL EXECUTION")
            if not self.export_tool_execution:
                for i, (prime, generated_content, tool_history, sampled_args) in renumerate(pending_tool_execution):
                    tool_id = tool_history[-1]["id"]
                    try:
                        parsed_args = self.arg_parsers[tool_id](self.decode(sampled_args))

                        logging.info(f"Executing tool {self.tool_names[tool_id]} with args {parsed_args}")
                        tool_output = self.tools[tool_id](*parsed_args)
                        tool_history[-1]["status"] = "Success"
                        tool_history[-1]["args"] = self.decode(sampled_args)
                        tool_history[-1]["parsed args"] = parsed_args
                        tool_history[-1]["output"] = tool_output

                    except Exception as e:
                        logging.warning(f"Error executing tool {self.tool_names[tool_id]} with args {parsed_args}")
                        # Print stack trace
                        logging.warning(traceback.format_exc())
                        logging.warning(f"Error: {e}")
                        tool_output = e
                        tool_history[-1]["status"] = "Error executing tool"

                        # Remove bad call from sentence
                        # sentence = sentence[:-sampled_args.shape[0]]
 
                    tool_output = self.encode(")→ " + str(tool_output), truncation=True, max_length=self.max_response_len)
                    tool_output = self.encode(self.decode(tool_output) + END_TOOL_TOKEN, return_tensors="pt")[0].to(device).long()
                    generated_content = torch.cat((generated_content[:-1], tool_output))

                    if generated_content.shape[0] < self.max_new_tokens:
                        pending_completion.append((prime, generated_content, tool_history))
                    else:                        
                        finished_sentences.append({"id":ids.pop(i), "original_prime":self.decode(prime.cpu()), "response":self.decode(generated_content.cpu()), "tool_history":tool_history})

        finished_sentences.sort(key=lambda x: x["id"])

        if self.export_tool_execution:
            return finished_sentences, pending_tool_execution

        
        return finished_sentences
    