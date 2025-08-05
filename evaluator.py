import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from model import GPSDRecommender
from data_preprocessing import SequenceDataset

class RecommendationEvaluator:
    """推荐系统评估器"""
    
    def __init__(self, model: GPSDRecommender, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def predict_top_k(self, 
                     input_ids: torch.Tensor, 
                     attention_mask: torch.Tensor,
                     k: int = 10,
                     exclude_seen: bool = True) -> torch.Tensor:
        """预测Top-K推荐物品"""
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            
            if exclude_seen:
                # 排除已经交互过的物品
                for i in range(input_ids.size(0)):
                    seen_items = input_ids[i][input_ids[i] != 0]  # 排除padding
                    logits[i, seen_items] = float('-inf')
            
            # 获取Top-K
            _, top_k_items = torch.topk(logits, k, dim=1)
            
        return top_k_items
    
    def calculate_recall_at_k(self, 
                             predictions: List[List[int]], 
                             ground_truth: List[List[int]], 
                             k: int = 10) -> float:
        """计算Recall@K"""
        total_recall = 0
        valid_users = 0
        
        for pred, truth in zip(predictions, ground_truth):
            if len(truth) == 0:
                continue
                
            pred_set = set(pred[:k])
            truth_set = set(truth)
            
            recall = len(pred_set & truth_set) / len(truth_set)
            total_recall += recall
            valid_users += 1
        
        return total_recall / valid_users if valid_users > 0 else 0.0
    
    def calculate_precision_at_k(self, 
                                predictions: List[List[int]], 
                                ground_truth: List[List[int]], 
                                k: int = 10) -> float:
        """计算Precision@K"""
        total_precision = 0
        valid_users = 0
        
        for pred, truth in zip(predictions, ground_truth):
            if len(truth) == 0:
                continue
                
            pred_set = set(pred[:k])
            truth_set = set(truth)
            
            if len(pred_set) > 0:
                precision = len(pred_set & truth_set) / len(pred_set)
                total_precision += precision
                valid_users += 1
        
        return total_precision / valid_users if valid_users > 0 else 0.0
    
    def calculate_ndcg_at_k(self, 
                           predictions: List[List[int]], 
                           ground_truth: List[List[int]], 
                           k: int = 10) -> float:
        """计算NDCG@K"""
        def dcg_at_k(relevances: List[float], k: int) -> float:
            """计算DCG@K"""
            dcg = 0
            for i, rel in enumerate(relevances[:k]):
                dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0
            return dcg
        
        total_ndcg = 0
        valid_users = 0
        
        for pred, truth in zip(predictions, ground_truth):
            if len(truth) == 0:
                continue
                
            truth_set = set(truth)
            
            # 计算预测结果的相关性
            relevances = [1.0 if item in truth_set else 0.0 for item in pred[:k]]
            
            # 计算DCG
            dcg = dcg_at_k(relevances, k)
            
            # 计算IDCG（理想DCG）
            ideal_relevances = [1.0] * min(len(truth), k)
            idcg = dcg_at_k(ideal_relevances, k)
            
            # 计算NDCG
            ndcg = dcg / idcg if idcg > 0 else 0.0
            total_ndcg += ndcg
            valid_users += 1
        
        return total_ndcg / valid_users if valid_users > 0 else 0.0
    
    def calculate_hit_rate_at_k(self, 
                               predictions: List[List[int]], 
                               ground_truth: List[List[int]], 
                               k: int = 10) -> float:
        """计算Hit Rate@K"""
        hits = 0
        valid_users = 0
        
        for pred, truth in zip(predictions, ground_truth):
            if len(truth) == 0:
                continue
                
            pred_set = set(pred[:k])
            truth_set = set(truth)
            
            if len(pred_set & truth_set) > 0:
                hits += 1
            valid_users += 1
        
        return hits / valid_users if valid_users > 0 else 0.0
    
    def calculate_coverage(self, 
                          predictions: List[List[int]], 
                          total_items: int,
                          k: int = 10) -> float:
        """计算推荐覆盖率"""
        recommended_items = set()
        
        for pred in predictions:
            recommended_items.update(pred[:k])
        
        return len(recommended_items) / total_items
    
    def calculate_diversity(self, 
                           predictions: List[List[int]], 
                           item_features: Optional[Dict[int, List]] = None,
                           k: int = 10) -> float:
        """计算推荐多样性（基于物品特征的余弦相似度）"""
        if item_features is None:
            # 如果没有物品特征，使用简单的多样性度量
            total_diversity = 0
            valid_users = 0
            
            for pred in predictions:
                if len(pred) < 2:
                    continue
                    
                # 计算推荐列表中物品的唯一性
                unique_items = len(set(pred[:k]))
                diversity = unique_items / min(len(pred), k)
                total_diversity += diversity
                valid_users += 1
            
            return total_diversity / valid_users if valid_users > 0 else 0.0
        
        # 基于物品特征的多样性计算
        # 这里可以扩展实现更复杂的多样性度量
        return 0.0
    
    def evaluate_model(self, 
                      test_sequences: List[Tuple[List[int], int]],
                      k_values: List[int] = [5, 10, 20],
                      batch_size: int = 64) -> Dict[str, float]:
        """全面评估模型性能"""
        
        print("开始模型评估...")
        
        # 准备测试数据
        test_dataset = SequenceDataset(test_sequences, self.model.backbone.max_seq_len)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        all_predictions = []
        all_ground_truth = []
        
        # 生成预测
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="生成预测"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                targets = batch['target'].cpu().numpy()
                
                # 预测Top-K
                max_k = max(k_values)
                top_k_items = self.predict_top_k(input_ids, attention_mask, k=max_k)
                
                # 收集预测和真实标签
                for i in range(len(targets)):
                    pred = top_k_items[i].cpu().numpy().tolist()
                    truth = [targets[i]] if targets[i] != 0 else []  # 排除padding
                    
                    all_predictions.append(pred)
                    all_ground_truth.append(truth)
        
        # 计算各种指标
        results = {}
        
        for k in k_values:
            results[f'Recall@{k}'] = self.calculate_recall_at_k(all_predictions, all_ground_truth, k)
            results[f'Precision@{k}'] = self.calculate_precision_at_k(all_predictions, all_ground_truth, k)
            results[f'NDCG@{k}'] = self.calculate_ndcg_at_k(all_predictions, all_ground_truth, k)
            results[f'HitRate@{k}'] = self.calculate_hit_rate_at_k(all_predictions, all_ground_truth, k)
        
        # 计算覆盖率和多样性
        results['Coverage@10'] = self.calculate_coverage(all_predictions, self.model.n_items, k=10)
        results['Diversity@10'] = self.calculate_diversity(all_predictions, k=10)
        
        return results
    
    def compare_models(self, 
                      baseline_results: Dict[str, float],
                      current_results: Dict[str, float]) -> Dict[str, float]:
        """比较模型性能"""
        improvements = {}
        
        for metric in current_results:
            if metric in baseline_results:
                baseline_val = baseline_results[metric]
                current_val = current_results[metric]
                
                if baseline_val != 0:
                    improvement = (current_val - baseline_val) / baseline_val * 100
                    improvements[f'{metric}_improvement_%'] = improvement
                else:
                    improvements[f'{metric}_improvement_%'] = float('inf') if current_val > 0 else 0
        
        return improvements
    
    def simulate_ab_test(self, 
                        test_sequences: List[Tuple[List[int], int]],
                        baseline_gmv: float = 1000.0,
                        expected_improvement: float = 7.97) -> Dict[str, float]:
        """模拟A/B测试GMV提升"""
        
        # 评估模型性能
        results = self.evaluate_model(test_sequences)
        
        # 基于Recall@10估算GMV提升
        recall_10 = results.get('Recall@10', 0)
        
        # 简化的GMV提升模型：假设Recall提升直接转化为GMV提升
        # 实际情况中需要更复杂的业务模型
        estimated_gmv_improvement = recall_10 * expected_improvement
        
        new_gmv = baseline_gmv * (1 + estimated_gmv_improvement / 100)
        
        ab_test_results = {
            'baseline_gmv': baseline_gmv,
            'new_gmv': new_gmv,
            'gmv_improvement_%': estimated_gmv_improvement,
            'recall@10': recall_10,
            'expected_improvement_%': expected_improvement
        }
        
        return ab_test_results
    
    def plot_evaluation_results(self, 
                               results: Dict[str, float],
                               save_path: str = "./results/evaluation_results.png"):
        """绘制评估结果"""
        
        # 分离不同类型的指标
        recall_metrics = {k: v for k, v in results.items() if 'Recall' in k}
        precision_metrics = {k: v for k, v in results.items() if 'Precision' in k}
        ndcg_metrics = {k: v for k, v in results.items() if 'NDCG' in k}
        other_metrics = {k: v for k, v in results.items() 
                        if not any(x in k for x in ['Recall', 'Precision', 'NDCG'])}
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Recall指标
        if recall_metrics:
            k_values = [int(k.split('@')[1]) for k in recall_metrics.keys()]
            recall_values = list(recall_metrics.values())
            axes[0, 0].plot(k_values, recall_values, 'o-', color='blue', linewidth=2, markersize=8)
            axes[0, 0].set_title('Recall@K', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('K')
            axes[0, 0].set_ylabel('Recall')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_ylim(0, max(recall_values) * 1.1)
        
        # NDCG指标
        if ndcg_metrics:
            k_values = [int(k.split('@')[1]) for k in ndcg_metrics.keys()]
            ndcg_values = list(ndcg_metrics.values())
            axes[0, 1].plot(k_values, ndcg_values, 'o-', color='green', linewidth=2, markersize=8)
            axes[0, 1].set_title('NDCG@K', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('K')
            axes[0, 1].set_ylabel('NDCG')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(0, max(ndcg_values) * 1.1)
        
        # Precision指标
        if precision_metrics:
            k_values = [int(k.split('@')[1]) for k in precision_metrics.keys()]
            precision_values = list(precision_metrics.values())
            axes[1, 0].plot(k_values, precision_values, 'o-', color='red', linewidth=2, markersize=8)
            axes[1, 0].set_title('Precision@K', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('K')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_ylim(0, max(precision_values) * 1.1)
        
        # 其他指标
        if other_metrics:
            metric_names = list(other_metrics.keys())
            metric_values = list(other_metrics.values())
            
            bars = axes[1, 1].bar(range(len(metric_names)), metric_values, 
                                 color=['orange', 'purple', 'brown', 'pink'][:len(metric_names)])
            axes[1, 1].set_title('Other Metrics', fontsize=14, fontweight='bold')
            axes[1, 1].set_xticks(range(len(metric_names)))
            axes[1, 1].set_xticklabels(metric_names, rotation=45, ha='right')
            axes[1, 1].set_ylabel('Value')
            
            # 添加数值标签
            for bar, value in zip(bars, metric_values):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图片
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"评估结果图表已保存到: {save_path}")

def print_evaluation_report(results: Dict[str, float], 
                           ab_test_results: Optional[Dict[str, float]] = None):
    """打印评估报告"""
    
    print("\n" + "="*50)
    print("           推荐系统评估报告")
    print("="*50)
    
    # 离线指标
    print("\n📊 离线评估指标:")
    print("-" * 30)
    
    for metric, value in results.items():
        if '@' in metric:
            print(f"{metric:15s}: {value:.4f}")
        else:
            print(f"{metric:15s}: {value:.4f}")
    
    # A/B测试结果
    if ab_test_results:
        print("\n💰 A/B测试GMV模拟:")
        print("-" * 30)
        
        for metric, value in ab_test_results.items():
            if 'gmv' in metric.lower():
                if '%' in metric:
                    print(f"{metric:20s}: {value:.2f}%")
                else:
                    print(f"{metric:20s}: ${value:,.2f}")
            else:
                print(f"{metric:20s}: {value:.4f}")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    # 示例评估
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模拟模型和数据
    from model import GPSDRecommender
    
    model = GPSDRecommender(n_items=1000)
    evaluator = RecommendationEvaluator(model, device)
    
    # 创建测试序列
    test_sequences = [(list(range(1, 11)), 11) for _ in range(100)]
    
    # 评估模型
    results = evaluator.evaluate_model(test_sequences)
    
    # A/B测试模拟
    ab_results = evaluator.simulate_ab_test(test_sequences)
    
    # 打印报告
    print_evaluation_report(results, ab_results)
    
    # 绘制结果
    evaluator.plot_evaluation_results(results)