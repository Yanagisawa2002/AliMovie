#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速演示脚本
基于Transformer的冷启动推荐系统演示

使用方法:
python demo.py
"""

import torch
import numpy as np
import time
from typing import List, Dict

from data_preprocessing import MovieLensDataProcessor, create_dataloaders
from model import GPSDRecommender
from trainer import RecommenderTrainer
from evaluator import RecommendationEvaluator, print_evaluation_report

def create_demo_data(n_users: int = 1000, n_items: int = 500, n_interactions: int = 10000):
    """创建演示数据"""
    print("🎬 创建演示数据...")
    
    np.random.seed(42)
    
    # 生成用户-物品交互数据
    user_ids = np.random.randint(1, n_users + 1, n_interactions)
    item_ids = np.random.randint(1, n_items + 1, n_interactions)
    ratings = np.random.choice([3, 4, 5], n_interactions, p=[0.3, 0.4, 0.3])
    timestamps = np.random.randint(1000000000, 1600000000, n_interactions)
    
    # 创建DataFrame
    import pandas as pd
    data = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings,
        'timestamp': timestamps
    })
    
    # 去重并排序
    data = data.drop_duplicates(subset=['user_id', 'item_id'])
    data = data.sort_values(['user_id', 'timestamp'])
    
    print(f"✓ 生成 {len(data):,} 条交互记录")
    print(f"✓ 用户数: {data['user_id'].nunique():,}")
    print(f"✓ 物品数: {data['item_id'].nunique():,}")
    
    return data

def demo_data_preprocessing():
    """演示数据预处理"""
    print("\n" + "="*50)
    print("📊 数据预处理演示")
    print("="*50)
    
    # 创建处理器
    processor = MovieLensDataProcessor(max_seq_len=20)
    
    # 创建演示数据
    ratings = create_demo_data()
    
    # 编码用户和物品
    print("\n🔢 编码用户和物品ID...")
    ratings = processor.encode_items_and_users(ratings)
    
    # 识别长尾用户
    print("\n🎯 识别长尾用户...")
    long_tail_data = processor.identify_long_tail_users(ratings, threshold=3)
    
    # 创建预训练数据
    print("\n📦 创建预训练数据...")
    pretrain_data = processor.create_pretrain_data(ratings, sample_frac=0.2)
    
    # 生成序列
    print("\n🔄 生成训练序列...")
    pretrain_sequences = processor.create_sequences(pretrain_data)
    finetune_sequences = processor.create_sequences(long_tail_data)
    
    return processor, pretrain_sequences, finetune_sequences

def demo_model_architecture():
    """演示模型架构"""
    print("\n" + "="*50)
    print("🤖 模型架构演示")
    print("="*50)
    
    # 创建小型模型用于演示
    model = GPSDRecommender(
        n_items=500,
        d_model=128,
        n_heads=4,
        n_layers=3,
        d_ff=256,
        max_seq_len=20,
        dropout=0.1
    )
    
    print(f"🏗️ 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    batch_size = 8
    seq_len = 15
    
    input_ids = torch.randint(1, 501, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    print(f"📥 输入形状: {input_ids.shape}")
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    
    print(f"📤 输出形状: {logits.shape}")
    print(f"✓ 前向传播成功")
    
    # 演示冻结参数
    print("\n🔒 演示参数冻结:")
    print(f"预训练模式参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    model.finetune_mode()
    print(f"微调模式参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model

def demo_training(processor, pretrain_sequences, finetune_sequences, model):
    """演示训练过程"""
    print("\n" + "="*50)
    print("🔥 训练过程演示")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 使用设备: {device}")
    
    # 创建数据加载器
    print("\n🔧 创建数据加载器...")
    pretrain_train_loader, pretrain_val_loader = create_dataloaders(
        pretrain_sequences[:200], batch_size=16, max_seq_len=20  # 使用少量数据演示
    )
    
    finetune_train_loader, finetune_val_loader = create_dataloaders(
        finetune_sequences[:100], batch_size=16, max_seq_len=20
    )
    
    # 创建训练器
    trainer = RecommenderTrainer(model, device, learning_rate=1e-3)
    
    # 快速预训练演示
    print("\n⚡ 快速预训练演示 (1 epoch)...")
    start_time = time.time()
    
    pretrain_history = trainer.pretrain(
        pretrain_train_loader,
        pretrain_val_loader,
        epochs=1,
        save_path="./demo_pretrained.pth"
    )
    
    pretrain_time = time.time() - start_time
    print(f"✓ 预训练完成，耗时: {pretrain_time:.2f}秒")
    
    # 快速微调演示
    print("\n🎯 快速微调演示 (1 epoch)...")
    start_time = time.time()
    
    finetune_history = trainer.finetune(
        finetune_train_loader,
        finetune_val_loader,
        epochs=1,
        save_path="./demo_finetuned.pth",
        early_stopping_patience=1
    )
    
    finetune_time = time.time() - start_time
    print(f"✓ 微调完成，耗时: {finetune_time:.2f}秒")
    
    return trainer

def demo_evaluation(trainer, finetune_sequences):
    """演示评估过程"""
    print("\n" + "="*50)
    print("📈 评估过程演示")
    print("="*50)
    
    # 创建评估器
    evaluator = RecommendationEvaluator(trainer.model, trainer.device)
    
    # 使用少量测试数据
    test_sequences = finetune_sequences[-50:]  # 使用最后50个序列作为测试
    
    print(f"🧪 测试序列数量: {len(test_sequences)}")
    
    # 评估模型
    print("\n🔍 开始评估...")
    start_time = time.time()
    
    results = evaluator.evaluate_model(
        test_sequences,
        k_values=[5, 10],
        batch_size=16
    )
    
    evaluation_time = time.time() - start_time
    print(f"✓ 评估完成，耗时: {evaluation_time:.2f}秒")
    
    # A/B测试模拟
    print("\n💰 A/B测试模拟...")
    ab_test_results = evaluator.simulate_ab_test(
        test_sequences,
        baseline_gmv=100000.0,  # 10万基准GMV
        expected_improvement=7.97
    )
    
    # 打印结果
    print_evaluation_report(results, ab_test_results)
    
    return results, ab_test_results

def demo_recommendation_example(trainer, processor):
    """演示推荐示例"""
    print("\n" + "="*50)
    print("🎯 推荐示例演示")
    print("="*50)
    
    # 创建示例用户序列
    example_sequence = [1, 15, 23, 45, 67]  # 示例物品序列
    
    print(f"👤 用户历史交互: {example_sequence}")
    
    # 准备输入
    input_ids = torch.tensor([example_sequence + [0] * (20 - len(example_sequence))]).long()
    attention_mask = torch.tensor([[1] * len(example_sequence) + [0] * (20 - len(example_sequence))]).long()
    
    # 生成推荐
    trainer.model.eval()
    with torch.no_grad():
        logits = trainer.model(input_ids.to(trainer.device), attention_mask.to(trainer.device))
        
        # 排除已交互物品
        for item in example_sequence:
            logits[0, item] = float('-inf')
        
        # 获取Top-10推荐
        _, top_items = torch.topk(logits[0], 10)
        recommendations = top_items.cpu().numpy().tolist()
    
    print(f"🎬 Top-10 推荐物品: {recommendations}")
    
    # 计算推荐分数
    scores = torch.softmax(logits[0], dim=0)
    rec_scores = [scores[item].item() for item in recommendations]
    
    print("\n📊 推荐详情:")
    for i, (item, score) in enumerate(zip(recommendations, rec_scores), 1):
        print(f"  {i:2d}. 物品 {item:3d} - 分数: {score:.4f}")

def main():
    """主演示函数"""
    print("🚀 基于Transformer的冷启动推荐系统 - 快速演示")
    print("" + "="*60)
    
    try:
        # 1. 数据预处理演示
        processor, pretrain_sequences, finetune_sequences = demo_data_preprocessing()
        
        # 2. 模型架构演示
        model = demo_model_architecture()
        
        # 3. 训练演示
        trainer = demo_training(processor, pretrain_sequences, finetune_sequences, model)
        
        # 4. 评估演示
        results, ab_results = demo_evaluation(trainer, finetune_sequences)
        
        # 5. 推荐示例
        demo_recommendation_example(trainer, processor)
        
        print("\n" + "="*60)
        print("🎉 演示完成！")
        print("\n💡 提示:")
        print("  - 运行 'python main.py --mode full' 进行完整训练")
        print("  - 查看 README.md 了解详细使用方法")
        print("  - 调整参数以适应您的数据集")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        print("\n🔧 故障排除建议:")
        print("  1. 检查依赖包是否正确安装")
        print("  2. 确认Python版本 >= 3.7")
        print("  3. 如果内存不足，可以减少数据量")
        raise

if __name__ == "__main__":
    main()