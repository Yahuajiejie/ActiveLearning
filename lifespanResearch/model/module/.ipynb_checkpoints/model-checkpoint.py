import torch
import torch.nn as nn
# 确保你能导入 LogitsConfig，如果不行请检查 import 路径
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
from esm.utils.sampling import _BatchedESMProteinTensor

class ESMCForMaskedLM(nn.Module):
    """
    ESMC 掩码语言模型封装
    """
    def __init__(self, esmc_model):
            super().__init__()
            self.esmc = esmc_model
            # 训练时通常不需要 LogitsConfig，因为我们直接取 forward 的结果
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
            """
            Args:
                input_ids: [Batch, Seq_Len] 对应 ESMC 的 sequence_tokens
                labels: [Batch, Seq_Len] 掩码标签
                attention_mask: [Batch, Seq_Len] (可选) 
                                注意：ESMC 内部通常根据 pad token 自动生成 mask。
            如果你的 collator 传进来了 mask，确保它是 bool 类型传给 sequence_id。
            Tensor 名称,角色,内容特征,目的
            input_ids,题目 (输入),"包含氨基酸索引、Mask 标记 (<mask>)、特殊标记 (<cls>, <eos>) 和填充 (<pad>)。",模型的输入。模型看这个残缺的序列，试图去猜被遮住的部分。
            labels,答案 (真值),包含被遮住位置的原始氨基酸索引，其他位置通常填 -100。,用于计算 Loss。模型并不知道这个 Tensor，只有 Loss 函数用它来和模型的预测结果对答案。
            attention_mask,视野控制,只有 0 和 1 (或 True/False)。1 代表有效数据，0 代表填充数据。,告诉 Attention 机制：忽略填充部分 (<pad>)，只计算有效氨基酸之间的关联。
            """
            
            # --- 修正点 1: 处理 Mask ---
            # ESMC 的 forward 接受 sequence_id 作为 mask (Boolean类型)
            # 如果 collator 给的是 0/1 整数 mask，需要转为 bool
            sequence_id = None
            if attention_mask is not None:
                sequence_id = attention_mask.bool()
            
            # --- 修正点 2: 直接调用 forward ---
            # 不要用 .logits()，也不需要包装成 _BatchedESMProteinTensor
            # 直接传 input_ids 给 sequence_tokens
            # ESMC 继承自 nn.Module，直接调用 self.esmc(...) 即可触发 forward 并保留梯度
            outputs = self.esmc(
                sequence_tokens=input_ids,
                sequence_id=sequence_id
            )
    
            # ESMC 的 forward 返回的是 ESMCOutput dataclass
            # 包含: sequence_logits, embeddings, hidden_states
            logits = outputs.sequence_logits  # [Batch, Seq_Len, Vocab_Size]
    
            loss = None
            if labels is not None:
                # --- 修正点 3: Loss 计算 ---
                # CrossEntropyLoss 期望输入是 [Batch, Class, Seq_Len]
                # 或者将 logits flatten 为 [Batch*Seq_Len, Vocab_Size]
                # 这里我们用 permute 调整维度
                loss = self.loss_fn(logits.permute(0, 2, 1), labels)
    
            return {"loss": loss, "logits": logits}