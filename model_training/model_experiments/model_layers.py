# Here I aim to explore the layers of the GPT-J model with the aim of freezing the lower layers.


import torch


from transformers import LlamaForCausalLM

cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache"
"""model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B",
            revision="float16",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,          # CANT HANDLE DEEPSPEED ZERO 3
            cache_dir=cache_dir 
        )"""



model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",
                                            torch_dtype=torch.float16,
                                            low_cpu_mem_usage=True,
                                            token="***REMOVED***",
                                            cache_dir=cache_dir)



print("Model named parameters")
layer_names = [name for name, p in model.named_parameters()]
print(layer_names)
for name, p in model.named_parameters():
    print(name)
    print(p.shape)


print(torch.cuda.get_device_properties(0))