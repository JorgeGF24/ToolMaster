from transformers import AutoModelForCausalLM
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live

import torch

cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache/"


model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B",
            revision="float16",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True, cache_dir=cache_dir
        )
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)