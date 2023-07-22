# In this class we define the ToolMaster class, which is the main class for the toolmaster model.

import torch.nn as nn
import torch
import torch.nn.functional as F

from beartype import beartype
from beartype.typing import List, Callable

PAD_TOKEN = 50402
ARROW_TOKEN = 39310
TOOL_TOKEN = 50400
END_API_TOKEN = 50401
OPEN_PARENTHESIS = 7
CLOSE_PARENTHESIS = 8


def log(t, eps=1e-20): return t.clamp(min=eps).log()


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1., dim=-1, eps=1e-10):
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
        tool_explanation_prompts: List[torch.Tensor],
        tokenized_tools: List[List[int]],

    ): 
        
        self.model = model

        longest_tool = max(len(tool) for tool in tokenized_tools)


        tool_selection_dict = {}

        def tree_maker(tree, token, id, depth):
            tokens = list(tree.keys())
            if token not in tokens:
                tree[token] = id
            else:
                if token == OPEN_PARENTHESIS:
                    print(f"Warning: tool {tokenized_tools[id]} is already in the tree")
                    return
                # Check if instance of dictionary:
                if not isinstance(tree[token], dict):
                    other_id = tree[token]
                    next_token = tokenized_tools[other_id][depth+1] if depth + 1 < len(tokenized_tools[other_id]) else OPEN_PARENTHESIS
                    tree[token] = {next_token: other_id}
                next_token = tokenized_tools[id][depth+1] if depth + 1 < len(tokenized_tools[id]) else OPEN_PARENTHESIS
                tree_maker(tree[token], next_token, id, depth + 1)

        for i, tool in enumerate(tokenized_tools):
            tree_maker(tool_selection_dict, tool[0], i, 0)


        self.tool_selection_dict = tool_selection_dict
        self.tool_explanation_prompts = tool_explanation_prompts
        self.tools = available_tools

    def generate(self, 
                 prompt: torch.Tensor, 
                 max_length: int = 100, 
                 temperature: float = 1.0, 
                 stop_token: int = 198):

        # Create num_return_sequences prompts
        prompt_len = prompt.shape[0]

        # pad the prompt to the max length
        output = F.pad(prompt, (max_length - prompt_len,), value=PAD_TOKEN)

        # Add batch dimension if not present
        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)

        for i in range(max_length):
            last_logits = self.model(output, use_cache = False).logits[0, prompt_len + i]

            # greedy sample (but could be made non-greedy)
            sampled = gumbel_sample(last_logits, temperature=temperature)

            output[0, prompt_len+i+1] = sampled

            # If any sampled are </API> thats a call thats done
            if sampled == TOOL_TOKEN:
                next_logits = self.model(output, use_cache = False).logits[0, prompt_len + i + 1:]

                for i in range(len(self.tool_selection_dict)):
                    tokens = list(self.tool_selection_dict[i].keys())

                    if len(tokens) != set(tokens):
                        raise ValueError("Tokens must be unique")
                    tokens = torch.tensor().to(output.device)

                    scores = next_logits[0, :, tokens]

                    biggest_score = scores.max(dim=-1)[0]




                # From the first item in each tool, check which has the highest probability
                

                for finished_idx in finished_samples.nonzero().squeeze(1):
                    start = initial_positions[finished_idx,0]
                    end = position_indices[finished_idx,0]
                    # Args, api position, data index
                    data_index = data_indices[finished_idx].item()
                    call = GeneratedArgsCall(ArgString(output[finished_idx,start:end]), start.item() - prompt_size - api_call_len, data_index, call_counter)
                    call_counter += 1
                    generated_calls[str(data_index)] = generated_calls.get(str(data_index),[]) + [call]
                    generated_counter += 1
                    batch_indices = batch_indices[:-1]

                # Only continue generating for incomplete sequences 
                output = output[~finished_samples]
                initial_positions = initial_positions[~finished_samples]
                position_indices = position_indices[~finished_samples]
                data_indices = data_indices[~finished_samples]

            if generated_counter == batch_size:
                break

            # increment positions
            position_indices += 1


    def forward(self, x):
        # When we pass the input to the model, we get the output and the hidden state

        return self.model(x)
