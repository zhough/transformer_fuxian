import torch

def create_padding_mask(seq_ids, pad_token_id):
    """生成自注意力用的 Padding 掩码"""
    non_pad_mask = (seq_ids != pad_token_id)  # [B, seq_len]
    return non_pad_mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, seq_len]

def create_causal_mask(seq_len, device):
    """生成解码器自注意力用的因果掩码（屏蔽未来位置）"""
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0)  # [1, seq_len, seq_len]

def create_cross_attention_mask(tgt_ids, src_ids, pad_token_id):
    """生成交叉注意力用的掩码（目标→源）"""
    tgt_pad = (tgt_ids != pad_token_id).unsqueeze(1).unsqueeze(3)  # [B, 1, tgt_len, 1]
    src_pad = (src_ids != pad_token_id).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, src_len]
    return tgt_pad & src_pad  # [B, 1, tgt_len, src_len]

def create_cross_attention_mask1(tgt_mask,src_mask):
    tgt_mask = tgt_mask.unsqueeze(1).unsqueeze(3)
    src_mask = src_mask.unsqueeze(1).unsqueeze(2)
    return tgt_mask & src_mask

# class MyDataset(Dataset):
#     def __init__(self,src_data,tgt_data,tokenizer,max_seq_len=512):
#         self.src_data = src_data
#         self.tgt_data = tgt_data
#         self.tokenizer = tokenizer
#         self.max_seq_len = max_seq_len
#         self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.pad_token_id = self.tokenizer.pad_token_id
    
#     def __len__(self):
#         return len(self.src_data)

#     def __getitem__(self, idx):
#         src_text = self.src_data[idx]
#         tgt_text = self.tgt_data[idx]

#         # 对源序列编码（添加结束符，截断/填充到最大长度）
#         src_encoding = self.tokenizer(
#             src_text,
#             truncation=True,
#             max_length=self.max_seq_len,
#             padding="max_length",
#             return_tensors="pt"
#         )
#         src_ids = src_encoding["input_ids"].squeeze(0)  # [max_seq_len]

#         tgt_encoding = self.tokenizer(
#             tgt_text,
#             truncation=True,
#             max_length=self.max_seq_len,
#             padding='max_length',
#             return_tensors='pt'
#         )
#         tgt_ids = tgt_encoding['input_ids'].squeeze(0)

#         tgt_input = tgt_ids[:-1]
#         tgt_labels = tgt_ids[1:]

#         return {
#             'src_ids':src_ids,
#             'tgt_input':tgt_input,
#             'tgt_label':tgt_labels
#         }