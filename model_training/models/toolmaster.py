# In this class we define the ToolMaster class, which is the main class for the toolmaster model.

import os
import logging
import traceback

from functools import partial

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from beartype import beartype
from beartype.typing import List, Callable, Union, Dict

# Import parent class of AutoTokenizer
from transformers import LlamaTokenizer, AutoTokenizer, GPT2Tokenizer

pad_sequence = partial(pad_sequence, batch_first=True)

PAD_TOKEN = "[PAD]"
PAD_ID = -100
ARROW_TOKEN = 39310
TOOL_TOKEN_ID = 50400
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



@beartype
class ToolMaster(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        available_tools: List[Callable],
        arg_parsers: List[Callable],
        tool_explanation_prompts: Union[List[torch.Tensor], List[str]],
        tool_names: List[str],
        tool_short_desc: Dict,  # of the form "tool_name": "tool_short_desc"
        tokenizer: Union[LlamaTokenizer, AutoTokenizer, GPT2Tokenizer],
        debug_level: int = 0,
        log_dir: str = "/vol/bitbucket/jg2619/augmenting_llms/model_training/models/logs",
        export_tool_execution: bool = False,
    ): 
        super().__init__()

        global PAD_ID, PAD_TOKEN, OPEN_PARENTHESIS_ID, TOOL_TOKEN_ID, LOGIT_DISPLACEMENT
        
        self.model = model
        self.encode = tokenizer.encode
        self.decode = tokenizer.decode

        PAD_ID = tokenizer.pad_token_id
        PAD_TOKEN = tokenizer.pad_token

        OPEN_PARENTHESIS_ID = tokenizer.encode(PAD_TOKEN + OPEN_PARENTHESIS)[1:]

        tokenized_tools = [tokenizer.encode(tool_name) for tool_name in tool_names]
        self.tokenized_tools = tokenized_tools
        self.tool_names = tool_names

        tool_name_desc = [        ]
        for tool_name in tool_names:
            name_desc = tool_name
            if tool_name in tool_short_desc:
                name_desc += " (" + tool_short_desc[tool_name] +")"
            tool_name_desc.append(tool_name)

        self.available_tools_prompt = "You can use these tools to help you answer: " + ", ".join(tool_names) + ".\n\n"
        self.tokenized_available_tools_prompt = tokenizer.encode(self.available_tools_prompt)

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
        if isinstance(tool_explanation_prompts[0], str):
            tool_explanation_prompts = [tokenizer.encode(prompt) for prompt in tool_explanation_prompts]
        self.tool_explanation_prompts = tool_explanation_prompts
        self.tools = available_tools
        self.arg_parsers = arg_parsers
        self.tokenized_tools = tokenized_tools

        self.eos_token_id = tokenizer.eos_token_id
        self.debug_level = debug_level
        # Create log dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # count files in log dir
        i = len(os.listdir(log_dir))
        logging.basicConfig(filename=f'{log_dir}/{i}.log', level=logging.DEBUG if debug_level>0 else logging.INFO, format='%(asctime)s:  %(message)s', datefmt='%m/%d/%Y %I:%M:%S  ')
        print(f"Logging to {log_dir}/{i}.log")

        # Tokens with →
        self.arg_gen_stoppers = []
        for key, value in tokenizer.get_vocab().items():
            if "→" in key:
                self.arg_gen_stoppers.append(value)
        self.arg_gen_stoppers = torch.tensor(self.arg_gen_stoppers).to(self.model.device)

        self.export_tool_execution = export_tool_execution

        # COPY PROMPT X TIMES
        # self.tokenized_available_tools_prompt.unsqueeze(0).repeat(batch_size,1)

    @torch.no_grad()
    def generate(self, 
                 primes: List[torch.Tensor], 
                 prompts: Union[List[torch.Tensor], torch.Tensor],
                 tool_history: tuple[List[int],...] = None,
                 arg_selection_mode: bool = False,  # Arg selection mode VS free generation augmented with tool selection
                 batch_generated_count: torch.Tensor = None,
                 max_includes_tool_usage: bool = True,
                 max_new_tokens: int = 100,
                 temperature: float = 0.1, 
                 stop_tokens: Union[List[int],int,torch.Tensor] = 198,):

        device = self.model.device

        if isinstance(prompts, torch.Tensor):
            prompts = [prompts for _ in range(len(primes))]

        assert primes[0].dim() == 1, "Primes must be 1D tensor with the tokenized data"
        assert prompts[0].dim() == 1, "Prompt must be 1D tensor with the tokenized prompt"
        
        batch_size = len(primes)                                                        # BATCH SIZE
        prompt_lens = torch.tensor([prompt.shape[0] for prompt in prompts]).to(device)  # LENGTHS OF PREPENDED PROMPTS
        
        # Position of where to start generating for each row
        positions = torch.tensor([prime.shape[0] for prime in primes]).to(device).unsqueeze(1)
        positions += prompt_lens.unsqueeze(1)
        initial_positions = positions.clone()
        if batch_generated_count is None:     # Count of tokens generated for each row
            batch_generated_count = torch.zeros(batch_size, dtype=torch.int)
        if tool_history is None:        # History of tools used for each row
            tool_history = [[] for _ in range(batch_size)]
        generated_content = [None for _ in range(batch_size)]
        if not isinstance(stop_tokens, torch.Tensor):
            stop_tokens = torch.tensor([stop_tokens]).to(device).long().view(-1)

        batch_input = [torch.cat([prompt.to(device), prime.to(device)]) for prompt, prime in zip(prompts, primes)]
        batch_lengths = torch.tensor([row.shape[0] for row in batch_input]).to(device)
        batch_input = pad_sequence(batch_input, padding_value=PAD_ID)
        padding_count = (batch_lengths + max_new_tokens - batch_generated_count).max().item()
        batch_input = F.pad(batch_input, (0, padding_count,), value=PAD_ID)

        
        # Indexing tensor utils
        loop_to_data_idx = torch.arange(batch_size).to(device)                 # Mapping from loop index to batch index
        batch_indices = torch.arange(batch_size).to(device).unsqueeze(1)       # ARANGE THAT ADJUSTS TO THE LOOP BATCH SIZE AS SAMPLES FINISH
        
        # Tool selection utils
        loop_selection_depth = torch.zeros(batch_size).int().to(device)        # Depth of the tool selection tree
        loop_is_selecting_tools = torch.zeros(batch_size).bool().to(device)    # Indices where we are selecting a tool
        initial_opts = torch.tensor(list(self.tool_selection_dict.keys()))     # Initial tool options
        initial_opts = initial_opts.to(device).unsqueeze(1)
        current_opts = [initial_opts for _ in range(batch_size)]               # Current tool options for row

        count_done = 0
        loop_i = batch_generated_count.min().item()    # i is min value of generated_count:
        while loop_i < max_new_tokens and count_done < batch_size:
            # MODEL FORWARD CALL. MAINTAINS SHAPE EVEN AFTER INDEXING
            loop_last_logits= self.model(batch_input[loop_to_data_idx], use_cache=False).logits[batch_indices, positions[loop_to_data_idx] + LOGIT_DISPLACEMENT]

            if arg_selection_mode:   # Tool usage not available
                loop_last_logits[:, TOOL_TOKEN_ID] = 0

            # Gumbel sample for rows not selecting a tool. Tool selection has different sampling procedure
            sample_ids = loop_to_data_idx[~loop_is_selecting_tools]
            loop_sampled = torch.ones(batch_indices.shape[0], 1).long().to(device)*-1
            loop_sampled[~loop_is_selecting_tools] = gumbel_sample(loop_last_logits[~loop_is_selecting_tools], temperature=temperature)
            batch_input[sample_ids.unsqueeze(1), positions[sample_ids]] = loop_sampled
            batch_generated_count[sample_ids] += 1

            # Sampling procedure for rows selecting a tool
            if loop_is_selecting_tools.any():

                for loop_i in reversed(loop_is_selecting_tools.nonzero().squeeze(1)):
                    
                    data_i = loop_to_data_idx[loop_i].item()
                    # Tool names are composed of tokens. ie. [CAL] [CUL] [ATOR]. We call each token a syllable
                    # Options for the next syllable. 
                    syllable_opts = torch.tensor(list(current_opts[loop_i].keys())).to(device)
                    next_syllable_idx = loop_last_logits[loop_i,syllable_opts].argmax(dim=-1)
                    next_syllable = syllable_opts[next_syllable_idx].item()
                    batch_input[data_i, positions[loop_i]] = next_syllable
                    loop_selection_depth[loop_i] += 1
                    current_opts[data_i] = current_opts[data_i][next_syllable]

                    # If current opts is a dict, there is a tie between possible tools. We need to keep selecting syllables.
                    if not isinstance(current_opts[data_i], dict):   # ELSE: We've reached a tool id
                        tool_id = current_opts[data_i]
                        depth = loop_selection_depth[loop_i].item()+1   # Selection_depth = i means we've selected the ith syllable of tool name. Add 1 for indexing purposes
                        tool_len = len(self.tokenized_tools[tool_id])
                        batch_input[data_i, positions[data_i]-depth:positions[data_i]-depth+tool_len] = torch.tensor(self.tokenized_tools[tool_id]).to(device)
                        batch_input[data_i, positions[data_i]-depth+tool_len] = OPEN_PARENTHESIS_ID
                        batch_generated_count[data_i] += tool_len
                        generated_content[data_i] = batch_input[data_i, initial_positions[data_i]:positions[data_i]]

                        tool_history[data_i].append(tool_id)

                        # Remove index i
                        remove_index = torch.arange(loop_to_data_idx.shape[0]) != loop_i
                        loop_is_selecting_tools = loop_is_selecting_tools[remove_index]
                        loop_selection_depth = loop_selection_depth[remove_index]
                        loop_to_data_idx = loop_to_data_idx[remove_index]
                        loop_sampled = loop_sampled[remove_index]
                        batch_indices[:-1]

                        count_done += 1

            # Check if any row wants to use a tool
            just_sampled_tool = loop_sampled == TOOL_TOKEN_ID
            if (just_sampled_tool).any():   # New rows selecting tools!
                loop_is_selecting_tools[~loop_is_selecting_tools][just_sampled_tool] = True

            # Rows that reached the max number of tokens, we finish the call
            reached_limit = batch_generated_count[loop_to_data_idx] == max_new_tokens
            # Sequence that reached the stop token
            finished = torch.isin(loop_sampled.squeeze(1), stop_tokens) + reached_limit
            if finished.any():
                if not arg_selection_mode:
                    # These rows are done generating. Mark them as finished
                    positions[loop_to_data_idx[finished]] = -1

                for finished_i in finished.nonzero().squeeze(1):
                    data_i = loop_to_data_idx[finished_i].item()
                    generated_content[data_i] = batch_input[data_i, initial_positions[data_i]:positions[data_i]]
                    count_done += 1
                    if reached_limit[finished_i]:
                        logging.warn(f"Stopping generation at row {data_i} that reached the generation limit")
                        logging.warn(f"Data: {self.decode(batch_input[data_i])}")
                        if arg_selection_mode:
                            # model failed to generate arguments.
                            logging.warn(f"Model failed to generate arguments for: \ndata: {self.decode(batch_input[data_i])}")
                            logging.warn(f"Data id {data_i}")
                            logging.warn(f"Tool history: {tool_history[data_i]}")
                            positions[data_i] = -1    # This marks tool use error - rectifies use and resumes generation
                            generated_content[data_i] = torch.tensor([]).to(device).long()

                loop_to_data_idx = loop_to_data_idx[~finished]
                loop_is_selecting_tools = loop_is_selecting_tools[~finished]
                loop_selection_depth = loop_selection_depth[~finished]
                batch_indices = batch_indices[:-finished.sum().item()]

            loop_i += 1
            positions[loop_to_data_idx[batch_indices.view(-1)]] += 1

        # Return positions to their position in data
        finished_rows = positions == -1   # Rows with special code -1 have finished due to reaching the generation limit or due to tool use error
        positions[~finished_rows] -= prompt_lens[~finished_rows.squeeze(1)]
        if not max_includes_tool_usage:
            batch_generated_count[~finished_rows] -= torch.tensor([self.tokenized_tools[tool_history[i][-1]].shape[0] for i in range(batch_size)]).to(device)[~finished_rows] + 2 # +2 for open parenthesis and <TOOL> token

        output = {
            "output_sentences": [torch.cat((p.to(device), g)) for (p, g) in zip(primes, generated_content)],
            "positions": positions,
            "batch_generated_count": batch_generated_count,
            "tool_history": tool_history,
            "sampled_args": generated_content,
        }
        if not arg_selection_mode:
            del output["sampled_args"]

        return output


    def forward(self, 
                sentences: List[str],):

        # We receive a batch of texts. 
        logging.info("FORWARD TOOLMASTER")
        logging.info(f"Received batch of {len(sentences)} sentences")

        device = self.model.device

        # We tokenize the texts and store then in tuples with (tokenized_sentence, pos, count generation, tool_history)
        pending_completion = [(self.encode(sentence), 0, []) for sentence in sentences]
        finished_sentences = []

        while len(pending_completion) > 0:

            ####################################################
            # FREE GENERATION MODE AUGMENTED WITH TOOL SELECTION
            ####################################################

            i = 0
            batch_size = 11
            pending_arg_sampling = []
            pending_count = len(pending_completion)

            while pending_count > 0:
                logging.debug(f"Processing batch {i+1}. Sentences processed: {len(pending_completion)-pending_count}/{len(pending_completion)}   ({(len(pending_completion)-pending_count)/len(pending_completion)*100:.2f}%))")
                sentence_batch = pending_completion[pending_count-batch_size:pending_count]

                try:
                    primes,  gen_count, tool_history = zip(*sentence_batch)
                    output_dict = self.generate(primes = [torch.tensor(prime).to(device).long() for prime in primes],
                                                prompts = torch.tensor(self.tokenized_available_tools_prompt).to(device).long(),
                                                tool_history=tool_history,
                                                batch_generated_count=torch.tensor(gen_count).to(device),
                                                max_new_tokens = 100,
                                                max_includes_tool_usage = True,
                                                arg_selection_mode = False,
                                                stop_tokens=self.eos_token_id)            
                except torch.cuda.OutOfMemoryError as e: # type: ignore
                    batch_size-=5
                    sentence_batch = sentence_batch[5:]
                    logging.info(f"Out of memory error. Reducing batch size to {batch_size}")
                    continue
                
                pending_count -= batch_size      
                finished_count = 0
                tools_called = [0 for _ in range(len(self.tools))]
                for sentence, status, gen_count, tool_history in zip(*output_dict.values()):
                    if status == -1:
                        finished_sentences.append((sentence.cpu(), tool_history))
                        finished_count += 1
                        for tool_id in tool_history:
                            tools_called[tool_id] += 1
                    else:
                        pending_arg_sampling.append((sentence, gen_count, tool_history))

                logging.info(f"Batch {i+1} processed. Finished sentences: {finished_count}/{batch_size}, rest use tools.")
                logging.info(f"Tools were called the following number of times:")
                for tool_name, tool_count in zip(self.tool_names, tools_called):
                    logging.info(f"{tool_name}: {tool_count}")
                i+=1


            ####################################################
            # ARGUMENT GENERATION MODE
            ####################################################

            batch_size = 11
            pending_completion = []
            pending_tool_execution = []
            total_pending_args = len(pending_arg_sampling)
            pending_count = total_pending_args

            while pending_count > 0:
                logging.debug(f"Processing batch {i+1}. Sentences processed: {pending_count}/{total_pending_args}   ({pending_count/total_pending_args*100:.2f}%))")
                
                sentence_batch = pending_arg_sampling[pending_count-batch_size:pending_count]
                try:
                    sentences, gen_count, tool_histories = zip(*sentence_batch)
                    prompts = [self.tool_explanation_prompts[hist[-1]] for hist in tool_histories]
                    output_dict = self.generate(primes = sentences,
                                                prompts = torch.tensor(prompts).to(device).long(),
                                                tool_history=tool_histories,
                                                batch_generated_count=gen_count,
                                                max_new_tokens = 100,
                                                max_includes_tool_usage = True,
                                                arg_selection_mode = True,
                                                stop_tokens=self.arg_gen_stoppers)
                except torch.cuda.OutOfMemoryError as e: # type: ignore
                    batch_size-=5
                    sentence_batch = sentence_batch[5:]
                    logging.info(f"Out of memory error. Reducing batch size to {batch_size}")
                    continue
                
                pending_count -= batch_size
                finished_count = 0
                for i, (sentence, status, gen_count, tool_history, sampled_args) in enumerate(zip(*output_dict.values())):
                    if status == -1:
                        pending_completion.append((sentence, gen_count, tool_history))
                        finished_count += 1
                    else:
                        # TOOL SELECTION BABY
                        pending_tool_execution.append((sentence, gen_count, tool_history, sampled_args))

            ####################################################
            # TOOL EXECUTION
            ####################################################

            if not self.export_tool_execution:
                for sentence, gen_count, tool_history, sampled_args in pending_tool_execution:
                    tool_id = tool_history[-1]
                    try:
                        args = self.arg_parsers[tool_id](self.decode(sampled_args))

                        logging.info(f"Executing tool {self.tool_names[tool_id]} with args {args}")
                        tool_output = self.tools[tool_id](*args)
                        tool_output = self.encode(str(tool_output), return_tensors="pt").to(device).long()

                        sentence = torch.cat(sentence, tool_output.long())
                    except Exception as e:
                        logging.warn(f"Error executing tool {self.tool_names[tool_id]} with args {args}")
                        # Print stack trace
                        logging.warn(traceback.format_exc())
                        logging.warn(f"Error: {e}")
                        tool_output = "Error executing tool"

                        # Remove bad call from sentence
                        sentence = sentence[:-sampled_args.shape[0]]

                    pending_completion.append((sentence, gen_count, tool_history))




        if self.export_tool_execution:
            return finished_sentences, pending_tool_execution
        
        return finished_sentences
        



if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
    


    cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache"
    model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=cache_dir)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=cache_dir)

    toolmaster = ToolMaster(model,
                            available_tools = [lambda x: x, lambda x: x],
                            arg_parsers = [lambda x: x, lambda x: x],
                            tool_explanation_prompts = ["tool1", "tool2"],
                            tool_names = ["tool1", "tool2"],
                            tool_short_desc = {},
                            tokenizer = tokenizer,
                            debug_level = 1,
                            log_dir = "/vol/bitbucket/jg2619/augmenting_llms/model_training/models/logs",
                            export_tool_execution = False,
                            )
    
    sentences = ["This is a sentence", "This is another sentence"]
    output = toolmaster(sentences)
    print(output)
    print(toolmaster.decode(output[0][0]))

