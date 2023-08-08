# Test the generation capabilities of a trained model

from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
import torch


gpt2 = False
# Load the model and tokenizer
cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache/"
if gpt2:
    model = GPT2LMHeadModel.from_pretrained("/vol/bitbucket/jg2619/augmenting_llms/model_training/models/GPT2", cache_dir=cache_dir)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=cache_dir)
else:
    model = AutoModelForCausalLM.from_pretrained(
            "/vol/bitbucket/jg2619/augmenting_llms/model_training/models/GPT2",
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
        ).cuda()
    tokenizer = AutoTokenizer.from_pretrained("/vol/bitbucket/jg2619/augmenting_llms/model_training/models/tokenizer", truncate=True, max_length=270, cache_dir=cache_dir)



prompt = "Once upon a time, there were 100 kingdoms in the land of Foo. One day, 25% of them were destroyed by a dragon. How many kingdoms were left?"
prompt2 = "hey"

# Encode the prompt
encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").cuda()
encoded_prompt2 = tokenizer.encode(prompt2, add_special_tokens=False, return_tensors="pt").cuda()

# Generate the output
output = model.generate(encoded_prompt, max_length=100, do_sample=True, top_k=50, top_p=0.95, temperature=1.0, num_return_sequences=1)
print(output, flush=True)
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

print(decoded_output, flush=True)

# Generate the output
output = model.generate(encoded_prompt2, max_length=100, do_sample=True, top_k=50, top_p=0.95, temperature=1.0, num_return_sequences=1)
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

print(decoded_output, flush=True)

model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir=cache_dir)
output = model.generate(encoded_prompt, max_length=100, do_sample=True, top_k=50, top_p=0.95, temperature=1.0, num_return_sequences=1)

# Decode the output
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

print(decoded_output, flush=True)
output = model.generate(encoded_prompt2, max_length=100, do_sample=True, top_k=50, top_p=0.95, temperature=1.0, num_return_sequences=1)

# Decode the output
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

print(decoded_output, flush=True)