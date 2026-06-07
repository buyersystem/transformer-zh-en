"""
Transformer 训练脚本（论文原版实现）

注意：此脚本保留用于参考，推荐使用 train_llm.py 进行训练。
train_llm.py 提供了更好的性能：AMP混合精度 + CosineLR学习率调度 + AdamW优化器

用法: python train_2017.py
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
try:
    from torch.utils.tensorboard.writer import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TB_AVAILABLE = False
import os
import random
import numpy as np
from tqdm import tqdm

from config import get_args
from models.transformer import Transformer
from tokenizer import build_tokenizer
from dataset import TranslationDataset, collate_fn


def build_model(vocab_size, config):
    """构建Transformer模型"""
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=config.d_model,
        num_heads=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        d_ffn=config.d_ff,
        dropout=config.dropout,
        max_len=config.max_len,
        pad_idx=0
    )
    return model


def set_seed(seed):
    """
    设置随机种子，确保实验可复现
    
    【为什么需要设置种子？】
    深度学习涉及大量随机操作：
    - 参数初始化
    - Dropout
    - 数据增强
    
    固定种子可以保证每次训练结果一致
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr(step, d_model, warmup_steps):
    """
    计算学习率 - 论文原版公式
    
    【论文公式】
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """
    step = max(1, step)
    base_lr = d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
    return base_lr


def train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, epoch, args, global_step, writer):
    """
    训练一个epoch
    
    【训练步骤】
    1. 前向传播: model(src, tgt_input) → logits
    2. 计算损失: criterion(logits, tgt_output)
    3. 反向传播: loss.backward()
    4. 参数更新: optimizer.step()
    5. 学习率更新: scheduler.step()
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    # 梯度累积计数器
    # 【作用】将多个小batch的梯度累加，一次性更新
    # 【例如】batch_size=32, accumulate=2 → 有效batch=64
    accumulate_grad = getattr(args, 'accumulate_grad', 1)
    
    # 进度条 (只在主进程显示)
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}") if args.local_rank == 0 else train_loader
    
    for batch_idx, (src, tgt) in enumerate(pbar):
        src = src.to(device)
        tgt = tgt.to(device)
        
        # 【关键步骤】目标序列处理
        # tgt_input: 去掉最后一个token (作为输入)
        # tgt_output: 去掉第一个token (作为标签)
        # 
        # 示例: tgt = [<s>, hello, world, </s>]
        #        tgt_input = [<s>, hello, world]      # 去掉</s>
        #        tgt_output = [hello, world, </s>]     # 去掉<s>
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        # 前向传播
        logits = model(src, tgt_input)
        
        # 计算损失
        # 【关键注意点 - reshape vs view】
        # Transformer 中的 tensor 可能不连续，使用 .view() 会报错：
        # "RuntimeError: view size is not compatible with input tensor's size and stride"
        # 解决方案：使用 .reshape() 代替 .view()，reshape 可以处理非连续 tensor
        #
        # 将logits展平为(batch*tgt_len, vocab_size)
        # 将tgt_output展平为(batch*tgt_len,)
        # ignore_index=0 忽略padding位置的损失
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        
        # 梯度累积：缩放损失
        loss = loss / accumulate_grad
        
        # 反向传播
        loss.backward()
        
        # 梯度累积：达到累积步数后更新参数
        if (batch_idx + 1) % accumulate_grad == 0 or (batch_idx + 1) == len(train_loader):
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # TensorBoard: 记录训练步数和损失
            if args.local_rank == 0 and writer is not None:
                current_loss = loss.item() * accumulate_grad
                current_lr = scheduler.get_last_lr()[0]
                writer.add_scalar('Train/Loss', current_loss, global_step)
                writer.add_scalar('Train/Learning_Rate', current_lr, global_step)
            global_step += 1
        
        total_loss += loss.item() * accumulate_grad
        num_batches += 1
        
        # 日志输出
        if args.local_rank == 0 and batch_idx % args.log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                "loss": f"{loss.item() * accumulate_grad:.4f}", 
                "lr": f"{lr:.6f}"
            })
    
    return total_loss / num_batches, global_step


def evaluate(model, val_loader, criterion, device):
    """
    在验证集上评估模型
    
    【注意】
    - 不需要梯度计算，使用torch.no_grad()加速
    - 不使用梯度累积
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for src, tgt in val_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            logits = model(src, tgt_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    """
    主训练函数
    
    【训练流程】
    1. 解析参数
    2. 初始化DDP（如果是多卡）
    3. 构建分词器和数据集
    4. 构建模型
    5. 训练循环
    6. 保存模型
    """
    args = get_args()
    
    # ========== DDP初始化 ==========
    # 判断是否使用多卡训练
    # 单卡: RANK=-1, WORLD_SIZE=1
    # 多卡: RANK=0/1/2, WORLD_SIZE=3
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.rank = int(os.environ['RANK'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0
    
    # 初始化多卡训练环境
    if args.world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
    
    # 设置设备
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    
    # 设置随机种子
    set_seed(args.seed)
    
    if args.local_rank == 0:
        print(f"Training with {args.world_size} GPUs")
        print(f"Config: d_model={args.d_model}, nhead={args.nhead}, layers={args.num_encoder_layers}")
    
    # ========== 构建分词器 ==========
    # 首次运行会训练BPE（约2-3分钟），后续运行复用
    tokenizer = build_tokenizer(
        os.path.join(args.data_dir, "train.zh"),
        os.path.join(args.data_dir, "train.en"),
        args.vocab_size,
        os.path.join(args.checkpoint_dir, "bpe_unified")
    )
    
    # ========== 加载数据集 ==========
    train_dataset = TranslationDataset(args.data_dir, tokenizer, args.max_len, "train")
    val_dataset = TranslationDataset(args.data_dir, tokenizer, args.max_len, "valid")
    
    # 创建数据加载器
    if args.world_size > 1:
        # 分布式采样器：每个GPU处理数据的一部分
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            sampler=train_sampler, 
            collate_fn=collate_fn
        )
        val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, rank=args.rank)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            sampler=val_sampler, 
            collate_fn=collate_fn
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True
        )
    
    # ========== 构建模型 ==========
    model = build_model(len(tokenizer), args).to(device)
    
    # 包装为DDP模型
    if args.world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    
    # ========== 损失函数 ==========
    # 标签平滑：让模型不要过度自信，提高泛化能力
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_id,  # 忽略padding位置的损失
        label_smoothing=args.label_smoothing
    )
    
    # ========== 优化器 ==========
    # 论文推荐的 betas 和 eps 参数
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        betas=(0.9, 0.98), 
        eps=1e-9
    )
    
    # ========== 学习率调度器（论文原版 Warmup） ==========
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: get_lr(step, args.d_model, args.warmup_steps)
    )
    
    # ========== 断点续训 ==========
    start_epoch = 0
    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        if args.local_rank == 0:
            print(f"Loading checkpoint: {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location=device, weights_only=False)
        if args.world_size > 1:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # 创建检查点目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # ========== TensorBoard 初始化 ==========
    # 日志保存到 checkpoints/runs 目录
    # 启动 TensorBoard: tensorboard --logdir checkpoints/runs
    writer = None
    if args.local_rank == 0:
        if TB_AVAILABLE:
            log_dir = os.path.join(args.checkpoint_dir, "runs")
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir)
            print(f"TensorBoard logging to: {log_dir}")
        else:
            print("Warning: tensorboard not available, skipping logging")
    
    best_loss = float('inf')
    global_step = 0
    
    # ========== 训练循环 ==========
    for epoch in range(start_epoch, args.epochs):
        # DDP: 每个epoch需要设置随机种子
        if args.world_size > 1:
            train_sampler.set_epoch(epoch)
        
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, 
            criterion, device, epoch, args, global_step, writer
        )
        
        # 验证并保存
        if args.local_rank == 0:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}")
            
            val_loss = evaluate(
                model.module if args.world_size > 1 else model, 
                val_loader, criterion, device
            )
            print(f"Epoch {epoch}: val_loss={val_loss:.4f}")
            
            # TensorBoard: 记录验证损失
            if writer is not None:
                writer.add_scalar('Eval/Loss', val_loss, epoch)
                writer.add_scalar('Eval/train_loss', train_loss, epoch)
            
            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if args.world_size > 1 else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'tokenizer': tokenizer,
                    'args': args
                }, os.path.join(args.checkpoint_dir, "best_model.pt"))
                print(f"Saved best model with val_loss={val_loss:.4f}")
            
            # 定期保存检查点
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if args.world_size > 1 else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"))
    
    # 清理DDP环境
    if args.world_size > 1:
        dist.destroy_process_group()
    
    # 关闭 TensorBoard writer
    if writer is not None:
        writer.close()
    
    if args.local_rank == 0:
        print("Training completed!")
        print(f"View TensorBoard with: tensorboard --logdir {os.path.join(args.checkpoint_dir, 'runs')}")


if __name__ == "__main__":
    main()
