# Here I aim to explore the layers of the GPT-J model with the aim of freezing the lower layers.

from model_training.model_experiments.GPTJ_layers import GPTJ_LAYERS
import torch


from transformers import LlamaForCausalLM, AutoModelForCausalLM

LLAMA = True

cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache"
"""model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B",
            revision="float16",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,          # CANT HANDLE DEEPSPEED ZERO 3
            cache_dir=cache_dir 
        )"""


if LLAMA:
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",
                                                torch_dtype=torch.float16,
                                                low_cpu_mem_usage=True,
                                                token="***REMOVED***",
                                                cache_dir=cache_dir)
else:
    model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B",
            revision="float16",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,          # CANT HANDLE DEEPSPEED ZERO 3
            cache_dir=cache_dir 
        )


FROZEN_LAYERS = []
TOTAL_LAYERS = []
for i in range(0, 24):
    if i in [13, 19]: continue
    FROZEN_LAYERS += GPTJ_LAYERS[f"Layer {i}"]

print(FROZEN_LAYERS)

unfrozen_count = 0
unfrozen_size = 0
total_count = 0
total_size = 0

for name, p in model.named_parameters():
    print(name)

    if name not in FROZEN_LAYERS:
        unfrozen_count += p.numel()
        unfrozen_size += p.numel() * p.element_size()

    total_count += p.numel()
    total_size += p.numel() * p.element_size()


print(f"Unfrozen parameters: {unfrozen_count}")
print(f"Unfrozen size: {unfrozen_size}")

print(f"Total parameters: {total_count}")
print(f"Total size: {total_size}")

print(f"Percentage of unfrozen parameters: {unfrozen_count / total_count * 100}%")
print(f"Percentage of unfrozen size: {unfrozen_size / total_size * 100}%")

print(torch.cuda.get_device_properties(0))