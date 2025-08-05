import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from typing import List, Tuple, Dict

class MovieLensDataProcessor:
    """MovieLens-1M数据预处理器"""
    
    def __init__(self, data_path: str = "./data/", max_seq_len: int = 50):
        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.item_encoder = LabelEncoder()
        self.user_encoder = LabelEncoder()
        
    def load_movielens_data(self) -> pd.DataFrame:
        """加载MovieLens-1M数据集"""
        # 如果没有数据文件，创建模拟数据
        ratings_file = os.path.join(self.data_path, "ratings.dat")
        if not os.path.exists(ratings_file):
            print("创建模拟MovieLens数据...")
            return self._create_mock_data()
        
        # 加载真实数据
        ratings = pd.read_csv(
            ratings_file,
            sep="::",
            names=["user_id", "item_id", "rating", "timestamp"],
            engine="python"
        )
        return ratings
    
    def _create_mock_data(self) -> pd.DataFrame:
        """创建模拟MovieLens数据用于演示"""
        np.random.seed(42)
        
        # 创建用户和物品
        n_users = 6040
        n_items = 3952
        
        # 创建不同类型的用户：长尾用户(30%)和活跃用户(70%)
        long_tail_users = int(n_users * 0.3)  # 30%的长尾用户
        active_users = n_users - long_tail_users
        
        data_list = []
        
        # 为长尾用户生成1-4次交互
        for user_id in range(1, long_tail_users + 1):
            n_interactions = np.random.randint(1, 5)  # 1-4次交互
            item_ids = np.random.choice(range(1, n_items + 1), n_interactions, replace=False)
            for item_id in item_ids:
                data_list.append({
                    "user_id": user_id,
                    "item_id": item_id,
                    "rating": np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3]),
                    "timestamp": np.random.randint(978300760, 1046455200)
                })
        
        # 为活跃用户生成5-50次交互
        for user_id in range(long_tail_users + 1, n_users + 1):
            n_interactions = np.random.randint(5, 51)  # 5-50次交互
            item_ids = np.random.choice(range(1, n_items + 1), 
                                      min(n_interactions, n_items), replace=False)
            for item_id in item_ids:
                data_list.append({
                    "user_id": user_id,
                    "item_id": item_id,
                    "rating": np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3]),
                    "timestamp": np.random.randint(978300760, 1046455200)
                })
        
        # 创建DataFrame
        data = pd.DataFrame(data_list)
        
        # 去重并排序
        data = data.drop_duplicates(subset=["user_id", "item_id"])
        data = data.sort_values(["user_id", "timestamp"])
        
        return data
    
    def identify_long_tail_users(self, ratings: pd.DataFrame, threshold: int = 5) -> pd.DataFrame:
        """识别长尾用户（交互次数 < threshold）"""
        user_interaction_counts = ratings.groupby("user_id").size()
        long_tail_user_ids = user_interaction_counts[user_interaction_counts < threshold].index
        
        long_tail_data = ratings[ratings["user_id"].isin(long_tail_user_ids)]
        print(f"长尾用户数量: {len(long_tail_user_ids)}")
        print(f"长尾用户交互数量: {len(long_tail_data)}")
        
        return long_tail_data
    
    def create_pretrain_data(self, ratings: pd.DataFrame, sample_frac: float = 0.1) -> pd.DataFrame:
        """创建预训练数据（10%全体数据）"""
        pretrain_data = ratings.sample(frac=sample_frac, random_state=42)
        print(f"预训练数据量: {len(pretrain_data)}")
        return pretrain_data
    
    def encode_items_and_users(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """编码用户和物品ID"""
        ratings = ratings.copy()
        ratings["user_encoded"] = self.user_encoder.fit_transform(ratings["user_id"])
        ratings["item_encoded"] = self.item_encoder.fit_transform(ratings["item_id"])
        
        self.n_users = len(self.user_encoder.classes_)
        self.n_items = len(self.item_encoder.classes_)
        
        print(f"用户数量: {self.n_users}, 物品数量: {self.n_items}")
        return ratings
    
    def create_sequences(self, ratings: pd.DataFrame) -> List[Tuple[List[int], int]]:
        """创建序列数据：输入序列 -> 下一个物品"""
        sequences = []
        
        for user_id in ratings["user_encoded"].unique():
            user_data = ratings[ratings["user_encoded"] == user_id].sort_values("timestamp")
            items = user_data["item_encoded"].tolist()
            
            # 创建滑动窗口序列
            for i in range(1, len(items)):
                input_seq = items[:i]
                target_item = items[i]
                
                # 截断或填充序列
                if len(input_seq) > self.max_seq_len:
                    input_seq = input_seq[-self.max_seq_len:]
                
                sequences.append((input_seq, target_item))
        
        print(f"生成序列数量: {len(sequences)}")
        return sequences
    
    def save_encoders(self, save_path: str = "./models/"):
        """保存编码器"""
        os.makedirs(save_path, exist_ok=True)
        
        with open(os.path.join(save_path, "item_encoder.pkl"), "wb") as f:
            pickle.dump(self.item_encoder, f)
        
        with open(os.path.join(save_path, "user_encoder.pkl"), "wb") as f:
            pickle.dump(self.user_encoder, f)
        
        # 保存元数据
        metadata = {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "max_seq_len": self.max_seq_len
        }
        
        with open(os.path.join(save_path, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)

class SequenceDataset(Dataset):
    """序列数据集"""
    
    def __init__(self, sequences: List[Tuple[List[int], int]], max_seq_len: int = 50):
        self.sequences = sequences
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_seq, target = self.sequences[idx]
        
        # 填充序列
        padded_seq = [0] * (self.max_seq_len - len(input_seq)) + input_seq
        
        return {
            "input_ids": torch.tensor(padded_seq, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.long),
            "attention_mask": torch.tensor([0] * (self.max_seq_len - len(input_seq)) + [1] * len(input_seq), dtype=torch.long)
        }

def create_dataloaders(sequences: List[Tuple[List[int], int]], 
                      batch_size: int = 64, 
                      train_ratio: float = 0.8,
                      max_seq_len: int = 50) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器"""
    
    # 检查序列是否为空
    if len(sequences) == 0:
        print("警告: 序列为空，返回None数据加载器")
        return None, None
    
    # 如果序列太少，至少保证有一个样本用于验证
    if len(sequences) == 1:
        train_sequences = sequences
        val_sequences = sequences  # 使用同一个样本进行验证
    else:
        # 划分训练和验证集
        split_idx = max(1, int(len(sequences) * train_ratio))  # 至少保证有1个训练样本
        train_sequences = sequences[:split_idx]
        val_sequences = sequences[split_idx:] if split_idx < len(sequences) else sequences[-1:]  # 至少保证有1个验证样本
    
    # 创建数据集
    train_dataset = SequenceDataset(train_sequences, max_seq_len)
    val_dataset = SequenceDataset(val_sequences, max_seq_len)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_sequences)), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, len(val_sequences)), shuffle=False)
    
    return train_loader, val_loader

if __name__ == "__main__":
    # 示例使用
    processor = MovieLensDataProcessor()
    
    # 加载数据
    ratings = processor.load_movielens_data()
    
    # 编码
    ratings = processor.encode_items_and_users(ratings)
    
    # 识别长尾用户
    long_tail_data = processor.identify_long_tail_users(ratings)
    
    # 创建预训练数据
    pretrain_data = processor.create_pretrain_data(ratings)
    
    # 创建序列
    pretrain_sequences = processor.create_sequences(pretrain_data)
    finetune_sequences = processor.create_sequences(long_tail_data)
    
    # 保存编码器
    processor.save_encoders()
    
    print("数据预处理完成！")