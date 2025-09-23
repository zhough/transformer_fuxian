from model import Transformer
from train1 import process_dataset,Config
from transformers import GPT2Tokenizer, GPT2Model

config = Config()

def translate(model,src_text,tokenizer,config):
    
