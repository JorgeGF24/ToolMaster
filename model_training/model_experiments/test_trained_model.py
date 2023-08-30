# Test the generation capabilities of a trained model

from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, LlamaConfig
import torch

from torch.nn.utils.rnn import pad_sequence

MODEL = "GPTJ"
# Load the model and tokenizer
cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache/"
if MODEL == "gpt2":
    model = GPT2LMHeadModel.from_pretrained("/vol/bitbucket/jg2619/augmenting_llms/model_training/models/GPTJ_test", cache_dir=cache_dir)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=cache_dir)
elif MODEL == "GPTJ":
    model = AutoModelForCausalLM.from_pretrained(
            "/vol/bitbucket/jg2619/augmenting_llms/model_training/models/GPTJ_mickey",
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
        ).cuda()
    tokenizer = AutoTokenizer.from_pretrained("/vol/bitbucket/jg2619/augmenting_llms/model_training/models/tokenizer", truncate=True, max_length=270, cache_dir=cache_dir)
elif MODEL == "LLAMA":
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf",
                                                token="***REMOVED***",
                                                cache_dir=cache_dir)

    tokenizer.add_bos_token = False

    tokenizer.add_tokens(["[PAD]"])
    tokenizer.pad_token="[PAD]"
    
    config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf", 
                                            token="***REMOVED***",
                                            padding_idx=tokenizer.pad_token_id)
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",
                                                token="***REMOVED***",
                                                torch_dtype=torch.float16,
                                                low_cpu_mem_usage=True,
                                                config=config,
                                                cache_dir=cache_dir).cuda()
elif MODEL == "LLAMA-big":
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf",
                                                token="***REMOVED***",
                                                cache_dir=cache_dir)

    tokenizer.add_bos_token = False

    tokenizer.add_tokens(["[PAD]"])
    tokenizer.pad_token="[PAD]"
    
    config = LlamaConfig.from_pretrained("meta-llama/Llama-2-13b-hf", 
                                            token="***REMOVED***",
                                            padding_idx=tokenizer.pad_token_id)
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf",
                                                token="***REMOVED***",
                                                torch_dtype=torch.float16,
                                                low_cpu_mem_usage=True,
                                                config=config,
                                                cache_dir=cache_dir).cuda()

model.eval()

prompt = "Once upon a time, there were 100 kingdoms in the land of Foo. One day, 25% of them were destroyed by a dragon. How many kingdoms were left?"
prompt2 = "hey"
prompt3 = "WOW THIS IS THE COOLEST AND SERENDIPITEST THING THAT"
prompt4 = f"""Your task is to add calls to a Calculator API to a piece of text. The calls should help you get information required to complete the text. You can call the API by writing "[Calculator(expression)]" where "expression" is the expression to be computed. Here are some examples of API calls:

Input: Last year we collected 237342 apples, double of what we collected this year: 118671.
Output: Last year we collected 237342 apples, double of what we collected this year: [Calculator(237342/2)→118671] 118671.

Input: The number in the next term is 18 + 12 x 3 = 54.
Output: The number in the next term is 18 + 12 x 3 = [Calculator(18+12*3)→54] 54.

Input: A total of 252 matches were played, and 723 goals were scored (an average of 2.87 per match). This is twenty goals more than the 703 goals last year.
Output: A total of 252 matches were played, and 723 goals were scored (an average of [Calculator(723/252)→2.87] 2.87 per match). This is twenty goals more than the [Calculator(723-20)→703] 703 goals last year.

Input: I went to Paris in 1994 and stayed there until 2011, so in total, it was 17 years.
Output: I went to Paris in 1994 and stayed there until 2011, so in total, it was [Calculator(2011-1994)→17] 17 years.

Input: His 73.9 stroke average makes him not only third on the team, but it is 3.8 strokes less than his 2011–12 average of 77.7.
Output: His 73.9 stroke average makes him not only third on the team, but it is 3.8 strokes less than his 2011–12 average of [Calculator("""

prompt5 = f"""These are examples where we use results from a calculator tool to complete the sentence. The calls should help you get information required to complete the text. You can call the API by writing "[Calculator(expression)]" where "expression" is the expression to be computed. Here are some examples of API calls:

Example 1: Last year we collected 237342 apples, double of what we collected this year: [Calculator(237342/2)→118671] 118671.

Example 2: The number in the next term is 18 + 12 x 3 = [Calculator(18+12*3)→54] 54.

Example 3: A total of 252 matches were played, and 723 goals were scored (an average of [Calculator(723/252)→2.87] 2.87 per match). This is twenty goals more than in 2013, when the total was [Calculator(723-20)→703] 703 goals last year.

Example 4: I went to Paris in 1994 and stayed there until 2011, so in total, it was [Calculator(2011-1994)→17] 17 years.

Example 5: His 73.9 stroke average makes him not only third on the team, but it is 3.8 strokes less than his 2011–12 average of [Calculator("""

prompt6 = """These are the available tools: 
  - WikiSearch (searches Wikipedia)

Set up in 1954 as a merger of smaller groups, the Front de Libration Nationale fought a war for independence from France until 1962, when the French government signed a cease-fire agreement. The FLN became the only legal party in which country?"""

prompts = [prompt6]

# Encode the prompt
encoded_prompts = [tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")[0].cuda() for prompt in prompts]
print(f"Len of prompt: {encoded_prompts[0].shape[0]}")

input = pad_sequence(encoded_prompts, batch_first=True, padding_value=tokenizer.pad_token_id).cuda()

# Generate the output
output = model.generate(input, num_beams=10, max_new_tokens =100 )#do_sample=True, top_k=50, top_p=0.95, temperature=1.0, num_return_sequences=1)
decoded_output = [tokenizer.decode(out, skip_special_tokens=True) for out in output]

print(*decoded_output, flush=True)

print(torch.cuda.memory_summary(device=None, abbreviated=False))
print(torch.cuda.memory_summary())

raise Exception("Done")
# Generate the output
output = model.generate(encoded_prompt2, max_length=100, do_sample=True, top_k=50, top_p=0.95, temperature=1.0, num_return_sequences=1)
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

print(decoded_output, flush=True)

# Generate the output
output = model.generate(encoded_prompt3, max_length=100, do_sample=True, top_k=50, top_p=0.95, temperature=1.0, num_return_sequences=1)
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