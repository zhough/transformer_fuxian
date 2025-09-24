from datasets import load_dataset
from transformers import BertTokenizer
dataset = load_dataset("wmt19", "zh-en", split="train[:10]")
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
for sample in dataset['translation']:
    print(f"ZH: {sample['zh']}")
    print(f"EN: {sample['en']}")
    print(f"EN Tokens: {tokenizer.tokenize(sample['en'])}")
    print(f"EN Token IDs: {tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample['en']))}")