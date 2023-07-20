from transformers import AutoTokenizer

cache_dir = "/vol/bitbucket/jg2619/augmenting_llms/augmented_data_pipeline/toolformer/cache"
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir=cache_dir)

TOOL_START_TOKEN = "<TOOL>"
TOOL_END_TOKEN = "</TOOL>" 

tokenizer.add_tokens([TOOL_START_TOKEN, TOOL_END_TOKEN, "[PAD]"])
tokenizer.pad_token = "[PAD]"

tokenizer.save_pretrained("/vol/bitbucket/jg2619/models/tokenizer/")
