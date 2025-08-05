import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换并重塑为多头
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        attention_output, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask
        )
        
        # 合并多头
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.w_o(attention_output)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, V), attention_weights

class FeedForward(nn.Module):
    """前馈网络"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 自注意力
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TransformerRecommender(nn.Module):
    """基于Transformer的推荐模型（仅解码器）"""
    
    def __init__(self, 
                 n_items: int,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 d_ff: int = 1024,
                 max_seq_len: int = 50,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 嵌入层
        self.item_embedding = nn.Embedding(n_items + 1, d_model, padding_idx=0)  # +1 for padding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer解码器层
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 分类头（可冻结嵌入层后单独训练）
        self.classifier = nn.Linear(d_model, n_items)
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """创建因果掩码（下三角矩阵）"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    
    def create_padding_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """创建填充掩码"""
        # attention_mask: [batch_size, seq_len]
        # 返回: [batch_size, 1, seq_len, seq_len]
        batch_size, seq_len = attention_mask.shape
        
        # 扩展维度用于广播
        mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
        mask = mask.expand(batch_size, 1, seq_len, seq_len)
        
        return mask
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 嵌入
        x = self.item_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.positional_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # 创建掩码
        causal_mask = self.create_causal_mask(seq_len, device)
        
        if attention_mask is not None:
            padding_mask = self.create_padding_mask(attention_mask)
            # 合并因果掩码和填充掩码
            combined_mask = causal_mask * padding_mask
        else:
            combined_mask = causal_mask
        
        # 通过解码器层
        for layer in self.decoder_layers:
            x = layer(x, combined_mask)
        
        # 分类预测（只使用最后一个时间步）
        last_hidden = x[:, -1, :]  # [batch_size, d_model]
        logits = self.classifier(last_hidden)  # [batch_size, n_items]
        
        return logits
    
    def freeze_embeddings(self):
        """冻结嵌入层参数（微调阶段使用）"""
        self.item_embedding.weight.requires_grad = False
        
        for layer in self.decoder_layers:
            for param in layer.parameters():
                param.requires_grad = False
        
        print("已冻结嵌入层和Transformer层，仅训练分类头")
    
    def unfreeze_all(self):
        """解冻所有参数（预训练阶段使用）"""
        for param in self.parameters():
            param.requires_grad = True
        
        print("已解冻所有参数")
    
    def get_trainable_params(self):
        """获取可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class GPSDRecommender(nn.Module):
    """GPSD风格的推荐模型（生成式预训练 + 微调）"""
    
    def __init__(self, 
                 n_items: int,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 d_ff: int = 1024,
                 max_seq_len: int = 50,
                 dropout: float = 0.1):
        super().__init__()
        
        self.backbone = TransformerRecommender(
            n_items=n_items,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        self.n_items = n_items
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        return self.backbone(input_ids, attention_mask)
    
    def pretrain_mode(self):
        """设置为预训练模式"""
        self.backbone.unfreeze_all()
        self.train()
    
    def finetune_mode(self):
        """设置为微调模式"""
        self.backbone.freeze_embeddings()
        self.train()
    
    def save_pretrained(self, save_path: str):
        """保存预训练模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'n_items': self.n_items,
            'config': {
                'd_model': self.backbone.d_model,
                'max_seq_len': self.backbone.max_seq_len
            }
        }, save_path)
        print(f"模型已保存到: {save_path}")
    
    def load_pretrained(self, load_path: str):
        """加载预训练模型"""
        checkpoint = torch.load(load_path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型已从 {load_path} 加载")
        return checkpoint.get('config', {})

if __name__ == "__main__":
    # 测试模型
    n_items = 1000
    batch_size = 32
    seq_len = 20
    
    model = GPSDRecommender(n_items=n_items)
    
    # 创建测试数据
    input_ids = torch.randint(1, n_items + 1, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # 前向传播
    logits = model(input_ids, attention_mask)
    print(f"输出形状: {logits.shape}")  # [batch_size, n_items]
    
    # 测试冻结参数
    print(f"预训练模式可训练参数: {model.backbone.get_trainable_params()}")
    
    model.finetune_mode()
    print(f"微调模式可训练参数: {model.backbone.get_trainable_params()}")