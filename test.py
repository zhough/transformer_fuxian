from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from train1 import process_dataset,Config
config = Config()
dataset = load_dataset("wmt19", "zh-en", split="train[:4]",cache_dir=config.dataset_cache)
dataloader = DataLoader(dataset,batch_size=4,shuffle=False,num_workers=1)
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased",cache_dir=config.pretrained_cache) 
for data in dataloader:
    sample = data['translation']
    print(f"ZH: {sample['zh']}")
    print(f"EN: {sample['en']}")
    tokenized = process_dataset(sample,tokenizer)
    print(f'src_ids:{tokenized["src_ids"]}')
    print(f'src_mask:{tokenized["src_mask"]}')
    print(f'tgt_input:{tokenized["tgt_input"]}')
    print(f'tgt_mask:{tokenized["tgt_mask"]}')
