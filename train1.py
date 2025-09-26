import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm  # 用于显示进度条
import os
from model import Transformer
from utils import create_padding_mask, create_causal_mask, create_cross_attention_mask
import swanlab
from transformers import BertTokenizer,BertModel
from torch.amp import autocast, GradScaler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp

# 初始化分布式环境
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# 清理分布式环境
def cleanup():
    dist.destroy_process_group()

class Config():
    def __init__(self):
        self.embed_dim = 768
        self.num_heads = 8
        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
        self.hidden_dim = self.embed_dim *4
        self.max_seq_len = 512
        self.dropout = 0.1  
        self.epochs = 20
        self.batch_size = 16

        self.learning_rate = 3e-5
        self.weight_decay = 1e-4
        self.pad_token_id = 0
        self.eos_token_id = 102
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_save_path = './output1/transformer.pth'
        self.dataset_cache = './temp/dataset'
        self.pretrained_cache = './temp/models'
        self.swanlab_project_name = 'transformer-training_v6'
        self.best_model_path = './val_models/best_model.pth'
        self.temp_model = '/kaggle/input/transformer_v1/transformers/default/1/best_model.pth'
        self.step = 0
        # 新增分布式训练参数
        self.world_size = torch.cuda.device_count()
        self.dist_url = "env://"
        self.local_rank = -1

config = Config()



def init_model(tokenizer, pretrained_model=None,rank=0):
    # 初始化 Transformer 模型（复用 GPT2 的词嵌入和位置编码）
    model = Transformer(
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        vocab_size=tokenizer.vocab_size,
        pretrained_model=pretrained_model,  # 预训练词嵌入
        ffn_hidden_dim=config.hidden_dim,
        dropout=config.dropout
    )
    model.to(rank)
    # 2. 加载权重（关键：处理 module. 前缀）
    if hasattr(config, 'temp_model') and os.path.exists(config.temp_model):
        # 2.1 加载权重文件（指定 map_location 到当前 GPU）
        device = torch.device('cuda', rank)
        state_dict = torch.load(config.temp_model, map_location=device)
        
        # 2.2 移除所有权重键的 module. 前缀（核心修复步骤）
        new_state_dict = {}
        for key, value in state_dict.items():
            # 去掉键开头的 module.（若存在）
            if key.startswith('module.'):
                new_key = key[len('module.'):]  # 从第 7 个字符开始截取（module. 共 7 个字符）
            else:
                new_key = key  # 若没有前缀，直接保留原键
            new_state_dict[new_key] = value
        
        # 2.3 加载处理后的权重到模型
        model.load_state_dict(new_state_dict)
        print(f"成功加载模型权重（已移除 module. 前缀）到 GPU {rank}")
    # 使用DDP包装模型
    if config.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rank], find_unused_parameters=False
        )
    # 损失函数：忽略 padding 位置的损失
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_token_id).to(rank)

    # 优化器：使用 AdamW（带权重衰减的 Adam）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay  # 权重衰减，防止过拟合
    )
    # 学习率调度器：随训练步数衰减学习率
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,  # 周期为训练轮数
        eta_min=1e-6  # 最小学习率
    )
    scaler = GradScaler('cuda')
    return model, criterion, optimizer, scheduler ,scaler

def process_dataset(data,tokenizer,rank):
    '''输入一条源数据和一条翻译数据的字典'''  
    src_texts = data['zh']
    tgt_texts = data['en']
    src_tokenized = tokenizer(
        src_texts,
        #padding="max_length",
        padding=True,
        truncation=True,
        max_length=config.max_seq_len,
        return_tensors='pt',
        add_special_tokens=True
    )
    tgt_tokenized = tokenizer(
        tgt_texts,
        #padding="max_length",
        padding=True,
        truncation=True,
        max_length=config.max_seq_len,
        return_tensors="pt",
        add_special_tokens=True
    )
    src_ids = src_tokenized.input_ids.to(rank)
    src_mask = src_tokenized.attention_mask.to(rank)
    tgt_input = tgt_tokenized.input_ids[:, :-1].to(rank)  # 移除最后一个token
    tgt_label = tgt_tokenized.input_ids[:, 1:].to(rank)   # 移除第一个token
    tgt_mask = tgt_tokenized.attention_mask[:, :-1].to(rank)
    return {
        'src_ids':src_ids,
        'src_mask':src_mask,
        'tgt_input':tgt_input,
        'tgt_label':tgt_label,
        'tgt_mask':tgt_mask,
    }

def generate_sample(model, tokenizer, src_text, max_len=50,rank=0):
    if rank != 0:
        return ''
    
    model.eval()
    with torch.no_grad():
        src_encoding = tokenizer(
            src_text,
            padding="max_length",
            truncation=True,
            return_tensors='pt',
            max_length=config.max_seq_len,
        )
        src_ids = src_encoding["input_ids"].to(rank)
        tgt_ids = torch.tensor([[tokenizer.bos_token_id]], device=rank)
        for _ in range(max_len):

            logits = model(src_ids=src_ids, tgt_ids=tgt_ids)
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            tgt_ids = torch.cat([tgt_ids, next_token_id], dim=-1)
            if next_token_id.item() == tokenizer.eos_token_id:
                break
        return tokenizer.decode(tgt_ids[0], skip_special_tokens=True)

# --------------------------
# 4. 训练循环batch
# --------------------------
def train_epoch(model, tokenizer, dataloader, criterion, optimizer, scheduler, scaler, config,rank):
    model.train()  # 训练模式（启用 dropout 等）
    total_loss = 0.0
    # 遍历数据集
    for batch in tqdm(dataloader, desc="Training", disable=(rank != 0)):
        if rank == 0:
            config.step = config.step + 1
        data = process_dataset(batch['translation'],tokenizer,rank)
        src_ids = data['src_ids']
        tgt_input = data['tgt_input']
        tgt_label = data['tgt_label']
        with autocast('cuda'):
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
        scaler.scale(loss).backward()  # 梯度缩放，防止梯度爆炸
        # scaler.unscale_(optimizer)
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)  # 参数更新
        scaler.update()  # 梯度缩放更新

        #loss.backward()  # 计算梯度
        #optimizer.step()  # 更新参数
        if config.step % 100 == 0 and rank == 0:
            swanlab.log({
                f'step_loss': loss.item(),
            },step = config.step)

        total_loss += loss.item()
    scheduler.step()  # 学习率调度
    avg_loss = total_loss / len(dataloader)
    return {
        'avg_loss' : avg_loss,
    }

def validate(model,tokenizer, dataloader, criterion, config, rank):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader,desc="Validating", disable=(rank != 0)):
            data = process_dataset(batch['translation'],tokenizer,rank)
            src_ids = data['src_ids']
            tgt_input = data['tgt_input']
            tgt_label = data['tgt_label']
            logits = model(
                src_ids=src_ids,
                tgt_ids=tgt_input,
                pad_token_id=config.pad_token_id
            )
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_label.reshape(-1)
            )
            
            total_loss += loss.item()
        total_loss_tensor = torch.tensor(total_loss, device=rank)
        # 所有进程的 total_loss_tensor 求和（dist.all_reduce 会把结果存回每个进程的张量）
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        global_total_loss = total_loss_tensor.item()
        global_avg_loss = global_total_loss / (len(dataloader) * config.world_size)
        return {
            'avg_loss' : global_avg_loss,
        } 


def main(rank,world_size,config):
    setup(rank, world_size)
    if rank == 0:
        swanlab.login(api_key="Nj75sPpgjdzUONcpKxlg6")
        swanlab.init(
            project=config.swanlab_project_name,
            experiment_name="baseline-model",
            config=vars(config)  # 自动记录所有超参数
        )

    # 加载分词器（负责文本→子词ID）
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", cache_dir=config.pretrained_cache)
    # 加载预训练模型
    pretrained_model = BertModel.from_pretrained("bert-base-multilingual-cased",cache_dir=config.pretrained_cache)  
    tokenizer.eos_token = '[SEP]'
    tokenizer.bos_token = '[CLS]'
    tokenizer.pad_token = tokenizer.pad_token
    config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id  
    #加载翻译数据集
    train_dataset = load_dataset("wmt19", "zh-en", split="train[60000:120000]",cache_dir=config.dataset_cache)
    test_dataset = load_dataset("wmt19", "zh-en", split="validation[:10000]",cache_dir=config.dataset_cache)
    # 使用DistributedSampler进行数据分片
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    
    train_dataloader = DataLoader(train_dataset,batch_size=config.batch_size,
                                  num_workers=4,sampler=train_sampler,pin_memory=True,drop_last=True)
    test_dataloader = DataLoader(test_dataset,batch_size=config.batch_size,
                                 num_workers=4,sampler=test_sampler,pin_memory=True)
    
    if rank == 0:
        print("Sample training data:")
        for i in range(5):  # 打印前5条数据
            sample = train_dataset[i]['translation']
            print(f"Sample {i+1}:")
            print(f"ZH: {sample['zh']}")
            print(f"EN: {sample['en']}")
            print(f"ZH Tokens: {tokenizer.tokenize(sample['zh'])}")
            print(f"ZH Token IDs: {tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample['zh']))}")
            print(f"EN Tokens: {tokenizer.tokenize(sample['en'])}")
            print(f"EN Token IDs: {tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sample['en']))}")
            print()

    # 初始化模型、损失函数、优化器
    model, criterion, optimizer, scheduler, scaler = init_model(tokenizer,pretrained_model,rank)
    # 开始训练
    print(f"开始训练，设备：{config.device}")
    best_validate_loss = float('inf')
    for epoch in range(config.epochs):
        train_sampler.set_epoch(epoch)
        res = train_epoch(model, tokenizer,train_dataloader, criterion, optimizer, scheduler, scaler, config,rank)
        validate_res = validate(model,tokenizer,test_dataloader,criterion,config,rank)
        if rank == 0:
            avg_loss = res['avg_loss']
            validate_loss = validate_res['avg_loss']
            
            print(f"Epoch {epoch+1}/{config.epochs}, 平均损失：{avg_loss:.4f},validate损失:{validate_loss:.4f}")
            sample_text = '一开始，很多人把这次危机比作1982年或1973年所发生的情况，这样得类比是令人宽心的，因为这两段时期意味着典型的周期性衰退。'
            sample_output = generate_sample(model, tokenizer, sample_text)
            print(f'sample_output:{sample_output}')
            print('标准答案:At the start of the crisis, many people likened it to 1982 or 1973, which was reassuring, because both dates refer to classical cyclical downturns.')
            if best_validate_loss > validate_loss:
                torch.save(model.state_dict(), config.best_model_path)
                print(f'成功保存当前轮次模型参数到{config.best_model_path}')
                best_validate_loss = validate_loss

            swanlab.log({
                "train/epoch_avg_loss": avg_loss,  # 每轮平均损失
                'train/validate_loss':validate_loss,
            }, step=epoch + 1)  # 以 epoch 为步长
    if rank == 0:
        # 保存模型
        os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
        torch.save(model.state_dict(), config.model_save_path)
        print(f"模型已保存至：{config.model_save_path}")
    cleanup()

if __name__ == "__main__":
    mp.spawn(
        main,
        args=(config.world_size, config),
        nprocs=config.world_size,
        join=True
    )