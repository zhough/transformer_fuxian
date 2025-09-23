from transformers import GPT2Tokenizer, GPT2Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import create_padding_mask,create_causal_mask,create_cross_attention_mask


class PositionEncoding(nn.Module):
    def __init__(self,embed_dim) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        #self.vocab_size = vocab_size

    def forward(self,x):
        #x: [batch_size, seq_len, embed_dim]
        seq_len = x.shape[1]
        pos = torch.arange(0,seq_len,device=x.device).unsqueeze(1)
        div_term = torch.exp(-1*math.log(10000)*torch.arange(0,self.embed_dim,2)/self.embed_dim)
        pe = torch.zeros(seq_len,self.embed_dim,device=x.device)
        pe[:,0::2] = torch.sin(pos*div_term)
        pe[:,1::2] = torch.cos(pos*div_term)

        return x + pe.unsqueeze(0)
    
class SDPAttention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,q,k,v,mask=None):
        
        QK = torch.matmul(q,k.transpose(-2,-1))
        dk = q.size(-1) #等价shape[-1]
        attn_socre = QK/math.sqrt(dk)

        if mask is not None:
            attn_socre = attn_socre.masked_fill(mask==0,value=-1e9) #将mask = 0的部分填充value
        
        attn_weights = F.softmax(attn_socre,dim=-1) #[batch_size,seq_len_q,seq_len_k]
        output = torch.matmul(attn_weights,v)   #[batch_size, seq_len_q, head_dim]

        return output,attn_weights
    
class MultiHeadAttention(nn.Module):
    def __init__(self,embed_dim,num_heads:int=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim//self.num_heads

        #self.qkv_proj = nn.Linear(self.embed_dim,self.embed_dim*3)
        self.q_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.output_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.single_head_attention = SDPAttention()

    def forward(self,x,mask=None,encoder_kv=None):
        batch_size,seq_len,_ = x.shape
        # qkv = self.qkv_proj(x) #[batch_size,seq_len,embed_dim*3]
        # qkv = qkv.reshape(batch_size,seq_len,3,self.num_heads,self.head_dim)
        # q,k,v = qkv.unbind(2) #[batch_size,seq_len,num_heads,head_dim]  
        # q,k,v = q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)  #[batch_size,num_heads,seq_len,head_dim] 

        if encoder_kv is None:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
        else:
            q = self.q_proj(x)  #[batch_size,seq_len,embed_dim]
            k = self.k_proj(encoder_kv)
            v = self.v_proj(encoder_kv)
        #拆分为多头注意力
        q = q.reshape(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        k = k.reshape(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        v = v.reshape(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        #合并batch_size和num_heads
        q = q.reshape(-1,seq_len,self.head_dim)
        k = k.reshape(-1,seq_len,self.head_dim)
        v = v.reshape(-1,seq_len,self.head_dim)

        #重构建mask
        if mask is not None:    #[B, seq_len_q, seq_len_k]
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mask = mask.repeat(1,self.num_heads,1,1).reshape(-1,seq_len,seq_len)

        #并行计算多头注意力
        attn_output,_ = self.single_head_attention(q,k,v,mask)  # [B*H, seq_len, head_dim]
        attn_output = attn_output.reshape(batch_size, self.num_heads, seq_len, self.head_dim).transpose(1, 2)
        attn_output = attn_output.reshape(batch_size,seq_len,self.embed_dim)

        attn_output = self.output_proj(attn_output)
        
        return attn_output

class FFN(nn.Module):
    def __init__(self,embed_dim,hidden_dim=None,dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim if hidden_dim else self.embed_dim * 4
        
        #全连接层结构
        self.fc1 = nn.Linear(self.embed_dim,self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim,self.embed_dim)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        batch_size,seq_len,_ = x.shape
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x    


class Encoder(nn.Module):   #先层归一化再接attn和ffn
    def __init__(self,embed_dim,num_heads:int=8,hidden_dim=None,dropout=0.1):
        super().__init__()

        self.embed_dim = embed_dim 
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads  
        self.dropout = dropout  

        self.attn = MultiHeadAttention(embed_dim=self.embed_dim,num_heads=self.num_heads)
        self.ffn = FFN(embed_dim=self.embed_dim,hidden_dim=self.hidden_dim,dropout=self.dropout)

        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)

    def forward(self,x,mask=None):
        residual = x
        x = self.norm1(x)
        x = self.attn(x,mask=mask)
        x = self.dropout1(x)
        x = x + residual

        #ffn
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout2(x)
        x = x + residual    

        return x
    
class Decoder(nn.Module):
    def __init__(self,embed_dim,num_heads:int=8,hidden_dim=None,dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.attn = MultiHeadAttention(self.embed_dim,num_heads=self.num_heads)
        self.ffn = FFN(self.embed_dim,self.hidden_dim,self.dropout)
        self.cross_attn = MultiHeadAttention(self.embed_dim,self.num_heads)

        self.dropout1 = nn.Dropout(self.dropout)
        self.dropout2 = nn.Dropout(self.dropout)
        self.dropout3 = nn.Dropout(self.dropout)

        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.norm3 = nn.LayerNorm(self.embed_dim)

    def forward(self, x, self_attn_mask=None, cross_attn_mask=None, encoder_output=None):
        '''
        causal_mask : 因果掩码，屏蔽未来位置
        self_attn_mask : 目标序列的padding掩码
        cross_attn_mask : 目标序列和源序列的padding掩码
        '''
        #自注意力
        residual = x
        #combined_mask = causal_mask if self_attn_mask is None else causal_mask & self_attn_mask
        x = self.norm1(x)
        x = self.attn(x,self_attn_mask)
        x = self.dropout1(x)
        x = x + residual

        #编码器解码器注意力
        if encoder_output is not None:
            residual = x
            x = self.norm2(x)   
            x = self.cross_attn(x,mask=cross_attn_mask,encoder_kv=encoder_output)
            x = self.dropout2(x)
            x = x + residual

        #ffn
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = self.dropout3(x)
        x = x + residual   
        
        return x

class TransformerEncoder(nn.Module):
    def __init__(self,embed_dim,num_heads,num_layers,ffn_hidden_dim=None,dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            Encoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                hidden_dim=ffn_hidden_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        )
        
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self,x,mask=None):
        for layer in self.layers:
            x = layer(x,mask)
        x = self.norm(x)
        return x
    
class TransformerDecoder(nn.Module):
    def __init__(self,embed_dim,num_heads,num_layers,ffn_hidden_dim=None,dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            Decoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                hidden_dim=ffn_hidden_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x, self_attn_mask=None, cross_attn_mask=None,encoder_output=None):
        for layer in self.layers:
            x = layer(x, self_attn_mask=self_attn_mask, cross_attn_mask=cross_attn_mask,encoder_output=encoder_output)
        x = self.norm(x)
        return x

class Transformer(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 num_heads, 
                 num_encoder_layers, 
                 num_decoder_layers, 
                 vocab_size,
                 pretrained_wte=None,
                 pretrained_wpe=None,
                 ffn_hidden_dim=None, 
                 dropout=0.1):
        super().__init__()
        if (pretrained_wte is None) or (pretrained_wpe is None):
            self.word_embedding = nn.Embedding(vocab_size,embed_dim)
            self.position_encoding = PositionEncoding(embed_dim)
        else:
            self.word_embedding = pretrained_wte
            
            self.position_encoding = pretrained_wpe

        self.encoders = TransformerEncoder(embed_dim=embed_dim,
                                           num_heads=num_heads,
                                           num_layers=num_encoder_layers,
                                           ffn_hidden_dim=ffn_hidden_dim,
                                           dropout=dropout)
        self.decoders = TransformerDecoder(embed_dim=embed_dim,
                                           num_heads=num_heads,
                                           num_layers=num_decoder_layers,
                                           ffn_hidden_dim=ffn_hidden_dim,
                                           dropout=dropout)
        
        self.output_proj = nn.Linear(embed_dim,vocab_size)
        self._init_weights()

    def _init_weights(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)  # 适合线性层的初始化

    def forward(self, src_ids=None, tgt_ids=None, pad_token_id=0,src_pad_mask = None,tgt_pad_mask=None):

        encoder_output=None
        cross_attn_mask = None

        if src_ids is not None:
            src_word_ebd = self.word_embedding(src_ids)
            batch_size, src_seq_len = src_ids.shape
            src_pos = torch.arange(0, src_seq_len, device=src_ids.device)  # [src_seq_len]
            src_pos = src_pos.unsqueeze(0).repeat(batch_size, 1)  # [B, src_seq_len]（Int 类型）
            src_pos_ebd = self.position_encoding(src_pos)
            src_ebd = src_word_ebd + src_pos_ebd
            #编码器层
            src_mask = create_padding_mask(src_ids,pad_token_id) if src_pad_mask is None else src_pad_mask
            src_mask = src_mask.repeat(1, 1, src_ids.shape[1], 1)
            encoder_output = self.encoders(src_ebd,src_mask)
        if tgt_ids is not None:
            #解码器层
            tgt_word_emb = self.word_embedding(tgt_ids)  # [B, tgt_seq_len, embed_dim]
            batch_size, tgt_seq_len = tgt_ids.shape
            tgt_pos = torch.arange(0, tgt_seq_len, device=tgt_ids.device)  # [tgt_seq_len]
            tgt_pos = tgt_pos.unsqueeze(0).repeat(batch_size, 1)  # [B, tgt_seq_len]
            tgt_pos_emb = self.position_encoding(tgt_pos)  # [B, tgt_seq_len, embed_dim]
            tgt_emb =  tgt_word_emb + tgt_pos_emb
            tgt_pad_mask = create_padding_mask(tgt_ids,pad_token_id) if tgt_pad_mask is None else tgt_pad_mask
            tgt_pad_mask = tgt_pad_mask.repeat(1,1,tgt_ids.shape[1],1)# [B, 1, tgt_len, tgt_len]
            tgt_causal_mask = create_causal_mask(tgt_ids.shape[1],device=tgt_ids.device)
            combined_mask = torch.logical_and(tgt_pad_mask,tgt_causal_mask)

            #交叉注意力掩码
            cross_attn_mask = create_cross_attention_mask(tgt_ids, src_ids, pad_token_id)
            x = self.decoders(tgt_emb,combined_mask,cross_attn_mask,encoder_output)

            #输出
            logits = self.output_proj(x)
            return logits
        return encoder_output



        

        