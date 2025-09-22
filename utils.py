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

