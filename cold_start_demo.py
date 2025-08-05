#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
冷启动用户推荐演示脚本
展示如何为新用户或交互历史很少的用户生成推荐

使用方法:
python cold_start_demo.py
"""

import torch
import numpy as np
import pandas as pd
import pickle
import os
from typing import List, Dict, Tuple

from model import GPSDRecommender
from data_preprocessing import MovieLensDataProcessor

class ColdStartRecommender:
    """冷启动推荐器"""
    
    def __init__(self, model_path: str, models_dir: str = './models/'):
        """
        初始化冷启动推荐器
        
        Args:
            model_path: 训练好的模型路径
            models_dir: 模型文件夹路径
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_dir = models_dir
        
        # 加载模型配置
        self._load_metadata()
        
        # 创建并加载模型
        self.model = GPSDRecommender(
            n_items=self.n_items,
            d_model=self.model_config.get('d_model', 384),
            n_heads=self.model_config.get('n_heads', 12),
            n_layers=self.model_config.get('n_layers', 8),
            d_ff=self.model_config.get('d_ff', 1536),
            max_seq_len=self.model_config.get('max_seq_len', 80),
            dropout=self.model_config.get('dropout', 0.1)
        )
        
        self.model.load_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 加载编码器
        self._load_encoders()
        
        print(f"✓ 冷启动推荐器已初始化")
        print(f"✓ 设备: {self.device}")
        print(f"✓ 物品数量: {self.n_items}")
    
    def _load_metadata(self):
        """加载模型元数据"""
        metadata_path = os.path.join(self.models_dir, 'metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.n_items = metadata['n_items']
                self.n_users = metadata['n_users']
                self.model_config = metadata.get('model_config', {})
        else:
            # 默认配置
            self.n_items = 3952
            self.n_users = 6040
            self.model_config = {
                'd_model': 384,
                'n_heads': 12,
                'n_layers': 8,
                'd_ff': 1536,
                'max_seq_len': 80,
                'dropout': 0.1
            }
    
    def _load_encoders(self):
        """加载用户和物品编码器"""
        # 加载物品编码器
        item_encoder_path = os.path.join(self.models_dir, 'item_encoder.pkl')
        if os.path.exists(item_encoder_path):
            with open(item_encoder_path, 'rb') as f:
                self.item_encoder = pickle.load(f)
        else:
            self.item_encoder = None
            
        # 加载用户编码器
        user_encoder_path = os.path.join(self.models_dir, 'user_encoder.pkl')
        if os.path.exists(user_encoder_path):
            with open(user_encoder_path, 'rb') as f:
                self.user_encoder = pickle.load(f)
        else:
            self.user_encoder = None
    
    def recommend_for_new_user(self, 
                              user_preferences: List[int] = None,
                              k: int = 10,
                              strategy: str = 'popular') -> List[int]:
        """
        为全新用户生成推荐
        
        Args:
            user_preferences: 用户偏好物品列表（可选）
            k: 推荐物品数量
            strategy: 推荐策略 ('popular', 'random', 'model')
            
        Returns:
            推荐物品ID列表
        """
        if strategy == 'popular':
            # 基于流行度的推荐
            return self._recommend_popular_items(k)
        elif strategy == 'random':
            # 随机推荐
            return self._recommend_random_items(k)
        elif strategy == 'model' and user_preferences:
            # 基于模型的推荐
            return self._recommend_with_model(user_preferences, k)
        else:
            # 默认使用流行度推荐
            return self._recommend_popular_items(k)
    
    def recommend_for_cold_user(self, 
                               user_history: List[int],
                               k: int = 10) -> List[int]:
        """
        为冷启动用户（有少量历史交互）生成推荐
        
        Args:
            user_history: 用户历史交互物品列表
            k: 推荐物品数量
            
        Returns:
            推荐物品ID列表
        """
        return self._recommend_with_model(user_history, k)
    
    def _recommend_with_model(self, 
                             user_sequence: List[int],
                             k: int = 10) -> List[int]:
        """使用训练好的模型生成推荐"""
        max_seq_len = self.model.backbone.max_seq_len
        
        # 准备输入序列
        if len(user_sequence) > max_seq_len:
            # 如果序列太长，取最后max_seq_len个物品
            input_sequence = user_sequence[-max_seq_len:]
        else:
            # 如果序列太短，进行padding
            input_sequence = user_sequence + [0] * (max_seq_len - len(user_sequence))
        
        # 创建attention mask
        attention_mask = [1 if item != 0 else 0 for item in input_sequence]
        
        # 转换为tensor
        input_ids = torch.tensor([input_sequence], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(self.device)
        
        # 生成推荐
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            
            # 排除已交互物品
            for item in user_sequence:
                if 1 <= item <= self.n_items:
                    logits[0, item] = float('-inf')
            
            # 获取Top-K推荐
            _, top_items = torch.topk(logits[0], k)
            recommendations = top_items.cpu().numpy().tolist()
        
        return recommendations
    
    def _recommend_popular_items(self, k: int = 10) -> List[int]:
        """推荐热门物品（简单实现）"""
        # 这里简化实现，实际应该基于物品流行度统计
        popular_items = list(range(1, min(k + 1, self.n_items + 1)))
        return popular_items
    
    def _recommend_random_items(self, k: int = 10) -> List[int]:
        """随机推荐物品"""
        return np.random.choice(range(1, self.n_items + 1), k, replace=False).tolist()
    
    def get_recommendation_scores(self, 
                                 user_sequence: List[int],
                                 candidate_items: List[int] = None) -> Dict[int, float]:
        """
        获取推荐分数
        
        Args:
            user_sequence: 用户历史序列
            candidate_items: 候选物品列表（可选）
            
        Returns:
            物品ID到推荐分数的映射
        """
        max_seq_len = self.model.backbone.max_seq_len
        
        # 准备输入
        if len(user_sequence) > max_seq_len:
            input_sequence = user_sequence[-max_seq_len:]
        else:
            input_sequence = user_sequence + [0] * (max_seq_len - len(user_sequence))
        
        attention_mask = [1 if item != 0 else 0 for item in input_sequence]
        
        input_ids = torch.tensor([input_sequence], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            scores = torch.softmax(logits[0], dim=0)
        
        # 返回指定物品的分数
        if candidate_items:
            return {item: scores[item].item() for item in candidate_items if 1 <= item <= self.n_items}
        else:
            return {item: scores[item].item() for item in range(1, self.n_items + 1)}

def demo_cold_start_scenarios():
    """演示不同的冷启动场景"""
    print("🚀 冷启动推荐演示")
    print("=" * 50)
    
    # 检查模型文件
    model_path = './models/finetuned_model.pth'
    if not os.path.exists(model_path):
        model_path = './models/pretrained_model.pth'
    
    if not os.path.exists(model_path):
        print("❌ 未找到训练好的模型，请先运行训练")
        print("   运行: python main.py --mode full")
        return
    
    # 初始化推荐器
    recommender = ColdStartRecommender(model_path)
    
    print("\n📋 冷启动推荐场景演示")
    print("-" * 30)
    
    # 场景1：全新用户（无任何历史）
    print("\n🆕 场景1：全新用户推荐")
    new_user_recs = recommender.recommend_for_new_user(k=10, strategy='popular')
    print(f"推荐物品: {new_user_recs}")
    
    # 场景2：有少量偏好的新用户
    print("\n👤 场景2：有偏好的新用户")
    user_preferences = [1, 15, 23]  # 用户表示喜欢这些物品
    pref_recs = recommender.recommend_for_new_user(
        user_preferences=user_preferences, 
        k=10, 
        strategy='model'
    )
    print(f"用户偏好: {user_preferences}")
    print(f"推荐物品: {pref_recs}")
    
    # 场景3：冷启动用户（有1-4次交互）
    print("\n❄️ 场景3：冷启动用户（少量历史）")
    cold_user_history = [5, 12, 28, 45]  # 用户历史交互
    cold_recs = recommender.recommend_for_cold_user(cold_user_history, k=10)
    print(f"用户历史: {cold_user_history}")
    print(f"推荐物品: {cold_recs}")
    
    # 场景4：获取推荐分数
    print("\n📊 场景4：推荐分数分析")
    candidate_items = [1, 10, 20, 30, 40, 50]
    scores = recommender.get_recommendation_scores(
        cold_user_history, 
        candidate_items
    )
    print(f"候选物品推荐分数:")
    for item, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  物品 {item:2d}: {score:.6f}")
    
    print("\n💡 使用建议:")
    print("  1. 新用户：使用流行度推荐或基于注册信息的推荐")
    print("  2. 有偏好新用户：结合偏好信息使用模型推荐")
    print("  3. 冷启动用户：充分利用有限的历史交互")
    print("  4. 持续学习：随着用户交互增加，推荐会越来越准确")

def create_recommendation_api_example():
    """创建推荐API使用示例"""
    print("\n🔧 推荐API使用示例")
    print("-" * 30)
    
    code_example = '''
# 推荐API使用示例
from cold_start_demo import ColdStartRecommender

# 初始化推荐器
recommender = ColdStartRecommender('./models/finetuned_model.pth')

# 为新用户推荐
def recommend_for_new_user(user_id):
    """为新用户推荐"""
    recommendations = recommender.recommend_for_new_user(
        k=10, 
        strategy='popular'
    )
    return recommendations

# 为冷启动用户推荐
def recommend_for_cold_user(user_id, user_history):
    """为冷启动用户推荐"""
    recommendations = recommender.recommend_for_cold_user(
        user_history=user_history,
        k=10
    )
    return recommendations

# 获取推荐分数
def get_item_scores(user_history, candidate_items):
    """获取物品推荐分数"""
    scores = recommender.get_recommendation_scores(
        user_sequence=user_history,
        candidate_items=candidate_items
    )
    return scores
'''
    
    print(code_example)

if __name__ == "__main__":
    try:
        demo_cold_start_scenarios()
        create_recommendation_api_example()
        print("\n🎉 冷启动推荐演示完成！")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        print("\n🔧 故障排除建议:")
        print("  1. 确保已完成模型训练")
        print("  2. 检查模型文件路径")
        print("  3. 确认依赖包正确安装")
        raise