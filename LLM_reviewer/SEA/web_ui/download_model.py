from transformers import AutoModel, AutoTokenizer
import os

# model name in huggingface
model_name = 'ECNU-SEA/SEA-E'

# denote download path
path_to_save = os.getenv("SEA_MODEL_CACHE", "models/SEA")

# download model and tokenizer
model = AutoModel.from_pretrained(model_name, cache_dir=path_to_save)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=path_to_save)
