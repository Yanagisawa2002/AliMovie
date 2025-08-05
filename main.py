#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于Transformer的冷启动推荐系统
参考阿里GPSD框架：生成式预训练 + 微调

主要功能：
1. MovieLens-1M数据预处理
2. 两阶段训练（预训练 + 微调）
3. 冷启动用户推荐评估
4. A/B测试GMV模拟
"""

import os
import torch
import argparse
import json
import time
from datetime import datetime
from typing import Dict, Any

from data_preprocessing import MovieLensDataProcessor, create_dataloaders
from model import GPSDRecommender
from trainer import RecommenderTrainer, create_trainer_from_config
from evaluator import RecommendationEvaluator, print_evaluation_report

def setup_directories():
    """创建必要的目录结构"""
    directories = [
        './data',
        './models',
        './results',
        './logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ 创建目录: {directory}")

def save_config(config: Dict[str, Any], save_path: str):
    """保存配置文件"""
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"✓ 配置已保存到: {save_path}")

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    print(f"✓ 配置已从 {config_path} 加载")
    return config

def main():
    parser = argparse.ArgumentParser(description='基于Transformer的冷启动推荐系统')
    
    # 基本参数
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'full'], 
                       default='full', help='运行模式')
    parser.add_argument('--config', type=str, default='./config.json', 
                       help='配置文件路径')
    parser.add_argument('--data_path', type=str, default='./data/', 
                       help='数据路径')
    parser.add_argument('--model_path', type=str, default='./models/', 
                       help='模型保存路径')
    parser.add_argument('--results_path', type=str, default='./results/', 
                       help='结果保存路径')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=256, help='模型维度')
    parser.add_argument('--n_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--n_layers', type=int, default=6, help='Transformer层数')
    parser.add_argument('--d_ff', type=int, default=1024, help='前馈网络维度')
    parser.add_argument('--max_seq_len', type=int, default=50, help='最大序列长度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    
    # 训练参数
    parser.add_argument('--pretrain_epochs', type=int, default=1, help='预训练轮数')
    parser.add_argument('--finetune_epochs', type=int, default=2, help='微调轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    
    # 数据参数
    parser.add_argument('--long_tail_threshold', type=int, default=5, 
                       help='长尾用户交互阈值')
    parser.add_argument('--pretrain_sample_frac', type=float, default=0.1, 
                       help='预训练数据采样比例')
    parser.add_argument('--train_ratio', type=float, default=0.8, 
                       help='训练集比例')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='auto', 
                       help='计算设备 (auto/cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🚀 启动推荐系统训练")
    print(f"📱 使用设备: {device}")
    print(f"🎯 运行模式: {args.mode}")
    print(f"🌱 随机种子: {args.seed}")
    
    # 创建目录
    setup_directories()
    
    # 保存配置
    config = vars(args)
    config['timestamp'] = datetime.now().isoformat()
    config['device_name'] = str(device)
    save_config(config, os.path.join(args.results_path, 'config.json'))
    
    # 数据预处理
    print("\n" + "="*50)
    print("📊 数据预处理阶段")
    print("="*50)
    
    processor = MovieLensDataProcessor(
        data_path=args.data_path,
        max_seq_len=args.max_seq_len
    )
    
    # 加载和处理数据
    print("📥 加载MovieLens数据...")
    ratings = processor.load_movielens_data()
    print(f"✓ 原始数据量: {len(ratings):,} 条评分")
    
    # 编码用户和物品
    print("🔢 编码用户和物品ID...")
    ratings = processor.encode_items_and_users(ratings)
    
    # 识别长尾用户
    print(f"🎯 识别长尾用户 (交互次数 < {args.long_tail_threshold})...")
    long_tail_data = processor.identify_long_tail_users(ratings, args.long_tail_threshold)
    
    # 创建预训练数据
    print(f"📦 创建预训练数据 ({args.pretrain_sample_frac*100:.1f}% 采样)...")
    pretrain_data = processor.create_pretrain_data(ratings, args.pretrain_sample_frac)
    
    # 创建序列数据
    print("🔄 生成序列数据...")
    pretrain_sequences = processor.create_sequences(pretrain_data)
    finetune_sequences = processor.create_sequences(long_tail_data)
    
    # 保存编码器
    processor.save_encoders(args.model_path)
    
    # 创建数据加载器
    print("🔧 创建数据加载器...")
    pretrain_train_loader, pretrain_val_loader = create_dataloaders(
        pretrain_sequences, args.batch_size, args.train_ratio, args.max_seq_len
    )
    
    finetune_train_loader, finetune_val_loader = create_dataloaders(
        finetune_sequences, args.batch_size, args.train_ratio, args.max_seq_len
    )
    
    print(f"✓ 预训练数据: {len(pretrain_sequences):,} 序列")
    print(f"✓ 微调数据: {len(finetune_sequences):,} 序列")
    
    if args.mode in ['train', 'full']:
        # 模型训练
        print("\n" + "="*50)
        print("🤖 模型训练阶段")
        print("="*50)
        
        # 创建模型配置
        model_config = {
            'd_model': args.d_model,
            'n_heads': args.n_heads,
            'n_layers': args.n_layers,
            'd_ff': args.d_ff,
            'max_seq_len': args.max_seq_len,
            'dropout': args.dropout
        }
        
        # 创建训练器
        trainer = create_trainer_from_config(
            n_items=processor.n_items,
            device=device,
            model_config=model_config
        )
        
        print(f"🏗️ 模型参数量: {sum(p.numel() for p in trainer.model.parameters()):,}")
        
        # 预训练阶段
        print("\n🔥 开始预训练...")
        start_time = time.time()
        
        pretrain_history = trainer.pretrain(
            pretrain_train_loader,
            pretrain_val_loader,
            epochs=args.pretrain_epochs,
            save_path=os.path.join(args.model_path, 'pretrained_model.pth')
        )
        
        pretrain_time = time.time() - start_time
        print(f"✓ 预训练完成，耗时: {pretrain_time:.2f}秒")
        
        # 微调阶段
        if finetune_train_loader is not None and finetune_val_loader is not None:
            print("\n🎯 开始微调...")
            start_time = time.time()
            
            finetune_history = trainer.finetune(
                finetune_train_loader,
                finetune_val_loader,
                epochs=args.finetune_epochs,
                save_path=os.path.join(args.model_path, 'finetuned_model.pth'),
                early_stopping_patience=3
            )
            
            finetune_time = time.time() - start_time
            print(f"✓ 微调完成，耗时: {finetune_time:.2f}秒")
        else:
            print("\n⚠️ 跳过微调阶段：长尾用户数据为空")
            finetune_history = {'train_loss': [], 'val_loss': []}
            finetune_time = 0
        
        # 绘制训练历史
        trainer.plot_training_history(
            os.path.join(args.results_path, 'training_history.png')
        )
        
        # 保存训练历史
        training_history = {
            'pretrain': pretrain_history,
            'finetune': finetune_history,
            'pretrain_time': pretrain_time,
            'finetune_time': finetune_time
        }
        
        with open(os.path.join(args.results_path, 'training_history.json'), 'w') as f:
            json.dump(training_history, f, indent=2)
    
    if args.mode in ['evaluate', 'full']:
        # 模型评估
        print("\n" + "="*50)
        print("📈 模型评估阶段")
        print("="*50)
        
        # 加载最佳模型
        model_path = os.path.join(args.model_path, 'finetuned_model.pth')
        if not os.path.exists(model_path):
            model_path = os.path.join(args.model_path, 'pretrained_model.pth')
        
        if os.path.exists(model_path):
            # 创建评估器 - 使用与训练时相同的模型配置
            model_config = {
                'd_model': args.d_model,
                'n_heads': args.n_heads,
                'n_layers': args.n_layers,
                'd_ff': args.d_ff,
                'max_seq_len': args.max_seq_len,
                'dropout': args.dropout
            }
            model = GPSDRecommender(
                n_items=processor.n_items,
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                d_ff=args.d_ff,
                max_seq_len=args.max_seq_len,
                dropout=args.dropout
            )
            model.load_pretrained(model_path)
            evaluator = RecommendationEvaluator(model, device)
            
            print(f"📥 加载模型: {model_path}")
            
            # 创建测试数据（使用长尾用户数据的一部分）
            test_sequences = finetune_sequences[-len(finetune_sequences)//4:]  # 使用25%作为测试集
            
            print(f"🧪 测试序列数量: {len(test_sequences):,}")
            
            # 评估模型
            print("🔍 开始模型评估...")
            start_time = time.time()
            
            results = evaluator.evaluate_model(
                test_sequences,
                k_values=[5, 10, 20],
                batch_size=args.batch_size
            )
            
            evaluation_time = time.time() - start_time
            print(f"✓ 评估完成，耗时: {evaluation_time:.2f}秒")
            
            # A/B测试模拟
            print("💰 模拟A/B测试GMV提升...")
            ab_test_results = evaluator.simulate_ab_test(
                test_sequences,
                baseline_gmv=1000000.0,  # 100万基准GMV
                expected_improvement=7.97
            )
            
            # 打印评估报告
            print_evaluation_report(results, ab_test_results)
            
            # 绘制评估结果
            evaluator.plot_evaluation_results(
                results,
                os.path.join(args.results_path, 'evaluation_results.png')
            )
            
            # 保存评估结果
            evaluation_results = {
                'metrics': results,
                'ab_test': ab_test_results,
                'evaluation_time': evaluation_time,
                'test_sequences_count': len(test_sequences)
            }
            
            with open(os.path.join(args.results_path, 'evaluation_results.json'), 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            
            print(f"💾 评估结果已保存到: {args.results_path}")
        
        else:
            print("❌ 未找到训练好的模型，请先运行训练模式")
    
    # 生成最终报告
    print("\n" + "="*50)
    print("📋 生成最终报告")
    print("="*50)
    
    report_lines = [
        "# 基于Transformer的冷启动推荐系统 - 实验报告\n",
        f"**实验时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"**设备**: {device}\n",
        f"**模式**: {args.mode}\n\n",
        "## 数据统计\n",
        f"- 原始数据量: {len(ratings):,} 条评分\n",
        f"- 用户数量: {processor.n_users:,}\n",
        f"- 物品数量: {processor.n_items:,}\n",
        f"- 长尾用户数量: {len(long_tail_data['user_encoded'].unique()):,}\n",
        f"- 预训练序列: {len(pretrain_sequences):,}\n",
        f"- 微调序列: {len(finetune_sequences):,}\n\n",
        "## 模型配置\n",
        f"- 模型维度: {args.d_model}\n",
        f"- 注意力头数: {args.n_heads}\n",
        f"- Transformer层数: {args.n_layers}\n",
        f"- 最大序列长度: {args.max_seq_len}\n\n",
        "## 训练配置\n",
        f"- 预训练轮数: {args.pretrain_epochs}\n",
        f"- 微调轮数: {args.finetune_epochs}\n",
        f"- 批次大小: {args.batch_size}\n",
        f"- 学习率: {args.learning_rate}\n\n"
    ]
    
    # 如果有评估结果，添加到报告中
    if args.mode in ['evaluate', 'full'] and 'results' in locals():
        report_lines.extend([
            "## 评估结果\n",
            f"- Recall@10: {results.get('Recall@10', 0):.4f}\n",
            f"- NDCG@10: {results.get('NDCG@10', 0):.4f}\n",
            f"- Precision@10: {results.get('Precision@10', 0):.4f}\n",
            f"- Coverage@10: {results.get('Coverage@10', 0):.4f}\n\n",
            "## A/B测试模拟\n",
            f"- 预期GMV提升: {ab_test_results.get('gmv_improvement_%', 0):.2f}%\n",
            f"- 新GMV: ${ab_test_results.get('new_gmv', 0):,.2f}\n\n"
        ])
    
    report_lines.append("---\n实验完成 ✅")
    
    # 保存报告
    report_path = os.path.join(args.results_path, 'experiment_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    
    print(f"📄 实验报告已保存到: {report_path}")
    print("\n🎉 实验完成！")

if __name__ == "__main__":
    main()