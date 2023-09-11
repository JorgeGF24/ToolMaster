from transformers import AutoTokenizer, LlamaTokenizer

cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache"
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir=cache_dir)
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf",
                                                   token="***REMOVED***",
                                                   cache_dir=cache_dir)

TOOL_START_TOKEN = "<TOOL>"
TOOL_END_TOKEN = "</TOOL>" 

tokenizer.add_tokens([TOOL_START_TOKEN, TOOL_END_TOKEN])
#tokenizer.add_special_tokens({"pad_token":"[PAD]"})
#tokenizer.pad_token = "[PAD]"

tokenizer.save_pretrained("/vol/bitbucket/jg2619/augmenting_llms/model_training/models/tokenizer-llama/")
