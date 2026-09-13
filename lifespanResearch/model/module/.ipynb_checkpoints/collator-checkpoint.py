import torch
from pathlib import Path
from esm.models.esmc import ESMC
from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    ESMProteinError,
    LogitsConfig,
    LogitsOutput,
    ProteinType,
)
from esm.tokenization import get_esm3_model_tokenizers, get_esmc_model_tokenizers
from esm.utils.constants.esm3 import data_root
from typing import List, Tuple

def get_vocab_range(tokenizer):
    """
    根据 tokenizer 自动获取氨基酸的 ID 范围 (low, high)。
    用于 torch.randint(low, high, ...)
    """
    target_tokens = list("ACDEFGHIKLMNPQRSTVWY" + "BJOUXZ")
    valid_ids = []
    for token in target_tokens:
        # 获取 ID
        if hasattr(tokenizer, "convert_tokens_to_ids"):
             # HuggingFace 风格
            idx = tokenizer.convert_tokens_to_ids(token)
        elif hasattr(tokenizer, "vocab"):
            # 字典风格
            idx = tokenizer.vocab.get(token)
        else:
            raise ValueError("无法识别 tokenizer 的结构，请检查方法名")

        # 确保 ID 有效且不是 unk (通常 unk 是 3)
        if idx is not None and idx != tokenizer.unk_token_id:
            valid_ids.append(idx)
            
    if not valid_ids:
        raise ValueError("未找到任何有效的氨基酸 ID，请检查 tokenizer")

    low = min(valid_ids)
    high = max(valid_ids) + 1 
    
    if len(valid_ids) != (high - low):
        print(f"警告: 氨基酸 ID 在 {low} 到 {high-1} 之间不连续！建议使用方案 B。")

    return low, high


class ESMCDataCollator:
    """
    数据整理器 (Data Collator)
    负责将 Dataset 返回的字符串列表动态转换为带有 [MASK] 的 Tensor
    """
    def __init__(self, tokenizer, mask_ratio: float = 0.15, device='cpu'):
        self.tokenizer = tokenizer
        self.mask_ratio = mask_ratio
        self.device = device
        
        # 缓存 Token ID
        self.mask_token_id = tokenizer.mask_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token_id = getattr(tokenizer, 'cls_token_id', 0)
        self.eos_token_id = getattr(tokenizer, 'eos_token_id', 2)
        self.vocab_size = len(tokenizer)
        self.low, self.high = get_vocab_range(self.tokenizer)
        print("special id",self.mask_token_id,self.pad_token_id,self.cls_token_id,self.eos_token_id)
        print(f"自动计算的范围: low={self.low}, high={self.high}")

    def __call__(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        DataLoader 会调用这个函数来组装 batch
        """
        # 1. 拆包 Dataset 返回的元组 (seq, lifetime)
        sequences = [item[0] for item in batch]
        lifetimes = torch.tensor([item[1] for item in batch], dtype=torch.float32).to(self.device)

        # 2. Tokenizer 编码
        encoded = self.tokenizer.batch_encode_plus(
            sequences,
            add_special_tokens=True,
            padding=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        # 3. 创建 Labels (用于计算 Loss)
        labels = input_ids.clone()
        
        # 4. 生成 Mask 概率矩阵
        probability_matrix = torch.full(labels.shape, self.mask_ratio, device=self.device)
        
        # 排除特殊 Token
        special_tokens_mask = [self.pad_token_id, self.cls_token_id, self.eos_token_id]
        for special_id in special_tokens_mask:
            if special_id is not None:
                probability_matrix.masked_fill_(input_ids == special_id, value=0.0)
        
        # 5. 生成掩码索引
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # 未被掩码的地方不计算 Loss

        # 6. 应用 80-10-10 策略
        # 80% -> [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=self.device)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id

        # 10% -> Random Token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=self.device)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(low=self.low, high=self.high, size=labels.shape, dtype=torch.long, device=self.device)
        input_ids[indices_random] = random_words[indices_random]

        # 剩下的 10% 保持原样

        return input_ids, labels, attention_mask, lifetimes