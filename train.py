import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm  # 用于显示进度条
import os
from model import Transformer
from utils import create_padding_mask, create_causal_mask, create_cross_attention_mask
from transformers import GPT2Tokenizer, GPT2Model
import swanlab

swanlab.login(api_key="Nj75sPpgjdzUONcpKxlg6")
class Config():
    def __init__(self):
        self.embed_dim = 768
        self.num_heads = 8
        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
        self.hidden_dim = self.embed_dim *4
        self.dropout = 0.1
        self.epochs = 10
        self.batch_size = 32
        self.learning_rate = 5e-5
        self.pad_token_id = 0  # padding token ID
        self.eos_token_id = 50256  # 结束符 token ID（GPT2 中默认）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_save_path = './output/transformer.pth'

config = Config()

class MyDataset(Dataset):
    def __init__(self,src_data,tgt_data,tokenizer,max_seq_len=512):
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id
    
    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src_text = self.src_data[idx]
        tgt_text = self.tgt_data[idx]

        # 对源序列编码（添加结束符，截断/填充到最大长度）
        src_encoding = self.tokenizer(
            src_text,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length",
            return_tensors="pt"
        )
        src_ids = src_encoding["input_ids"].squeeze(0)  # [max_seq_len]

        tgt_encoding = self.tokenizer(
            tgt_text,
            truncation=True,
            max_length=self.max_seq_len,
            padding='max_length',
            return_tensors='pt'
        )
        tgt_ids = tgt_encoding['input_ids'].squeeze(0)

        tgt_input = tgt_ids[:-1]
        tgt_labels = tgt_ids[1:]

        return {
            'src_ids':src_ids,
            'tgt_input':tgt_input,
            'tgt_label':tgt_labels
        }

def init_model(tokenizer, pretrained_model):
    # 初始化 Transformer 模型（复用 GPT2 的词嵌入和位置编码）
    model = Transformer(
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        vocab_size=tokenizer.vocab_size,
        pretrained_wte=pretrained_model.wte,  # 预训练词嵌入
        pretrained_wpe=pretrained_model.wpe,  # 预训练位置编码
        ffn_hidden_dim=config.hidden_dim,
        dropout=config.dropout
    ).to(config.device)

    # 损失函数：忽略 padding 位置的损失
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)

    # 优化器：使用 AdamW（带权重衰减的 Adam）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01  # 权重衰减，防止过拟合
    )
    # 学习率调度器：随训练步数衰减学习率
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,  # 周期为训练轮数
        eta_min=1e-6  # 最小学习率
    )

    return model, criterion, optimizer, scheduler


# --------------------------
# 4. 训练循环
# --------------------------
def train_epoch(model, dataloader, criterion, optimizer, scheduler, config):
    model.train()  # 训练模式（启用 dropout 等）
    total_loss = 0.0

    # 遍历数据集
    for batch in tqdm(dataloader, desc="Training"):
        # 数据移到设备上
        src_ids = batch["src_ids"].to(config.device)
        tgt_input = batch["tgt_input"].to(config.device)
        tgt_label = batch["tgt_label"].to(config.device)

        # 前向传播：模型输出 logits
        logits = model(
            src_ids=src_ids,
            tgt_ids=tgt_input,
            pad_token_id=config.pad_token_id
        )  # [batch_size, tgt_seq_len-1, vocab_size]

        # 计算损失（需展平 logits 和标签）
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),  # [batch_size*(tgt_seq_len-1), vocab_size]
            tgt_label.reshape(-1)  # [batch_size*(tgt_seq_len-1)]
        )

        # 反向传播 + 参数更新
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 计算梯度
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，防止梯度爆炸
        optimizer.step()  # 更新参数
        scheduler.step()  # 学习率调度

        total_loss += loss.item()

    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def main():
    swanlab.init(
        project="transformer-training",
        experiment_name="baseline-model",
        config=vars(config)  # 自动记录所有超参数（config 是你的配置类实例）
    )

    model_dir = './models/'
    # 加载分词器（负责文本→子词ID）
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2",cache_dir=model_dir)
    # 加载预训练模型（我们只需要它的词嵌入层）
    pretrained_model = GPT2Model.from_pretrained("gpt2",cache_dir=model_dir)    

    # src_data = ["我爱机器学习", "Transformer 是一种强大的模型", "自然语言处理很有趣"]
    # tgt_data = ["I love machine learning", "Transformer is a powerful model", "Natural language processing is interesting"]
    # dataset = MyDataset(src_data,tgt_data,tokenizer,max_seq_len=32)
    dataset = load_dataset("wmt14", "zh-en", split="train[:10000]",cache_dir='./dataset')
    dataloader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=2)

    # 初始化模型、损失函数、优化器
    model, criterion, optimizer, scheduler = init_model(tokenizer, pretrained_model)
    # 开始训练
    print(f"开始训练，设备：{config.device}")
    for epoch in range(config.epochs):
        avg_loss = train_epoch(model, dataloader, criterion, optimizer, scheduler, config)
        print(f"Epoch {epoch+1}/{config.epochs}, 平均损失：{avg_loss:.4f}")
        swanlab.log({
            "train/epoch_avg_loss": avg_loss  # 每轮平均损失
        }, step=epoch + 1)  # 以 epoch 为步长
    # 保存模型
    os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
    torch.save(model.state_dict(), config.model_save_path)
    print(f"模型已保存至：{config.model_save_path}")

if __name__ == "__main__":
    main()