from model import Transformer
from train1 import process_dataset,Config
from transformers import GPT2Tokenizer, GPT2Model
from transformers import BertTokenizer
import torch.nn as nn
import torch

config = Config()
pretrained_cache = './temp/models'
model_path = './temp/output/transformer.pth'


def init_model(tokenizer, pretrained_model):
    # 初始化 Transformer 模型（复用 GPT2 的词嵌入和位置编码）
    model = Transformer(
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        vocab_size=tokenizer.vocab_size,
        pretrained_wte=None,  # 预训练词嵌入
        pretrained_wpe=None,  # 预训练位置编码
        ffn_hidden_dim=config.hidden_dim,
        dropout=config.dropout
    )
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  # 自动分发到所有可用GPU
    model.to(config.device)
    return model

def generate(model,src_text,tokenizer,max_len=10):
    model.eval()
    src_encoding = tokenizer(
        src_text,
        padding=True,
        truncation=True,
        return_tensors='pt',
    )
    src_ids = src_encoding["input_ids"].to(config.device)
    tgt_ids = torch.tensor([[tokenizer.bos_token_id]]).to(config.device)
    for _ in range(max_len):
        # 模型预测下一个token
        logits = model(src_ids=src_ids, tgt_ids=tgt_ids)
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tgt_ids = torch.cat([tgt_ids, next_token_id], dim=-1)

        # 遇到结束符则停止
        # if next_token_id.item() == tokenizer.eos_token_id:
        #     break
        print(tgt_ids)
    return tokenizer.decode(tgt_ids[0], skip_special_tokens=True)


    




if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", cache_dir=pretrained_cache)
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '[EOS]'})
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.cls_token

    pretrained_model = GPT2Model.from_pretrained("gpt2",cache_dir=pretrained_cache)
    model = init_model(tokenizer,pretrained_model)
    #model.load_state_dict(torch.load(model_path, map_location=config.device))
    src_text = "我爱自然语言处理"
    result = generate(model,src_text,tokenizer)
    print(tokenizer.tokenize(src_text))  
    print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(src_text)))  
    print(f'output:{result}')


# git config --global user.email "453854697@qq.com"
# git config --global user.name "zhough"