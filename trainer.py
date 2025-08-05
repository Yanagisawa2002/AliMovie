import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import GPSDRecommender
from data_preprocessing import create_dataloaders

class EarlyStopping:
    """早停策略"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience

class RecommenderTrainer:
    """推荐系统训练器"""
    
    def __init__(self, 
                 model: GPSDRecommender,
                 device: torch.device,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5):
        
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            targets = batch['target'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, targets)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                targets = batch['target'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, targets)
                
                total_loss += loss.item()
                
                # 计算准确率
                _, predicted = torch.max(logits.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def pretrain(self, 
                train_loader: DataLoader, 
                val_loader: DataLoader,
                epochs: int = 1,
                save_path: str = "./models/pretrained_model.pth") -> Dict[str, List[float]]:
        """预训练阶段"""
        
        print("=== 开始预训练阶段 ===")
        self.model.pretrain_mode()
        
        # 重新初始化优化器（因为参数可能发生变化）
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-4,
            weight_decay=1e-5
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            
            # 记录
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                self.model.save_pretrained(save_path)
                print(f"保存最佳预训练模型: {save_path}")
        
        print("=== 预训练完成 ===")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }
    
    def finetune(self, 
                train_loader: DataLoader, 
                val_loader: DataLoader,
                epochs: int = 2,
                save_path: str = "./models/finetuned_model.pth",
                early_stopping_patience: int = 3) -> Dict[str, List[float]]:
        """微调阶段"""
        
        print("=== 开始微调阶段 ===")
        self.model.finetune_mode()
        
        # 重新初始化优化器（只训练分类头）
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=5e-4,  # 微调使用更高的学习率
            weight_decay=1e-5
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=2, factor=0.5
        )
        
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        best_val_loss = float('inf')
        
        finetune_train_losses = []
        finetune_val_losses = []
        finetune_learning_rates = []
        
        for epoch in range(epochs):
            print(f"\nFine-tune Epoch {epoch + 1}/{epochs}")
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            
            # 记录
            finetune_train_losses.append(train_loss)
            finetune_val_losses.append(val_loss)
            finetune_learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                self.model.save_pretrained(save_path)
                print(f"保存最佳微调模型: {save_path}")
            
            # 早停检查
            if early_stopping(val_loss):
                print(f"早停触发，在第 {epoch + 1} 轮停止训练")
                break
        
        print("=== 微调完成 ===")
        
        return {
            'train_losses': finetune_train_losses,
            'val_losses': finetune_val_losses,
            'learning_rates': finetune_learning_rates
        }
    
    def plot_training_history(self, save_path: str = "./results/training_history.png"):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(self.val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 学习率曲线
        axes[0, 1].plot(self.learning_rates, color='green')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].grid(True)
        
        # 损失对比（最后10个epoch）
        if len(self.train_losses) > 10:
            recent_train = self.train_losses[-10:]
            recent_val = self.val_losses[-10:]
            epochs_recent = list(range(len(self.train_losses)-10, len(self.train_losses)))
            
            axes[1, 0].plot(epochs_recent, recent_train, label='Train Loss', color='blue')
            axes[1, 0].plot(epochs_recent, recent_val, label='Val Loss', color='red')
            axes[1, 0].set_title('Recent Training Progress (Last 10 Epochs)')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # 模型参数统计
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        axes[1, 1].bar(['Total Params', 'Trainable Params'], 
                      [total_params, trainable_params],
                      color=['lightblue', 'orange'])
        axes[1, 1].set_title('Model Parameters')
        axes[1, 1].set_ylabel('Number of Parameters')
        
        # 添加数值标签
        for i, v in enumerate([total_params, trainable_params]):
            axes[1, 1].text(i, v + max(total_params, trainable_params) * 0.01, 
                           f'{v:,}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图片
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"训练历史图表已保存到: {save_path}")

def create_trainer_from_config(n_items: int, 
                              device: torch.device,
                              model_config: Optional[Dict] = None) -> RecommenderTrainer:
    """根据配置创建训练器"""
    
    if model_config is None:
        model_config = {
            'd_model': 256,
            'n_heads': 8,
            'n_layers': 6,
            'd_ff': 1024,
            'max_seq_len': 50,
            'dropout': 0.1
        }
    
    model = GPSDRecommender(n_items=n_items, **model_config)
    trainer = RecommenderTrainer(model, device)
    
    return trainer

if __name__ == "__main__":
    # 示例训练流程
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模拟数据
    n_items = 1000
    sequences = [(list(range(1, 11)), 11) for _ in range(1000)]  # 模拟序列数据
    
    train_loader, val_loader = create_dataloaders(sequences, batch_size=32)
    
    # 创建训练器
    trainer = create_trainer_from_config(n_items, device)
    
    print(f"模型总参数量: {sum(p.numel() for p in trainer.model.parameters()):,}")
    
    # 预训练
    pretrain_history = trainer.pretrain(train_loader, val_loader, epochs=1)
    
    # 微调
    finetune_history = trainer.finetune(train_loader, val_loader, epochs=2)
    
    # 绘制训练历史
    trainer.plot_training_history()
    
    print("训练完成！")