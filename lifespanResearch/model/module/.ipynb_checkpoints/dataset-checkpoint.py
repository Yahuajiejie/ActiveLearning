import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import torch
from typing import List, Tuple


class ProteinDataset(Dataset):
    """
    健壮的蛋白质序列数据集
    逻辑：
    1. 序列 (seq) 必须存在 -> 如果缺失或为空，整行数据丢弃 (Drop)。
    2. 参数 (如 lifetime) 可选 -> 如果缺失，填充为 -1。
    """
    def __init__(self, csv_file: str, max_length: int = 1024):
        """
        Args:
            csv_file (str): CSV 文件路径
            max_length (int): 截断长度
        """
        self.max_length = max_length
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"找不到文件: {csv_file}")
        print(f"正在读取 CSV 文件: {csv_file} ...")
        try:
            # 1. 读取 CSV
            df = pd.read_csv(csv_file)
            
            # ================== 核心清洗逻辑 ==================
            
            # A. 检查并清洗 'seq' 列 (序列)
            if 'seq' not in df.columns:
                raise ValueError("CSV 中必须包含 'seq' 列！")
            
            # 将空白字符转为 NaN，方便统一 drop
            # 这一步是为了防止有些数据是空字符串 "" 而不是 null
            df['seq'] = df['seq'].astype(str).replace(r'^\s*$', np.nan, regex=True)
            
            # 记录原始数量
            original_len = len(df)
            
            # 丢弃 seq 为 NaN 的行
            df = df.dropna(subset=['seq'])
            
            dropped_len = original_len - len(df)
            if dropped_len > 0:
                print(f"警告: 已丢弃 {dropped_len} 条 'seq' 为空的数据。")

            # B. 检查并清洗 'lifetime' 列 (参数)
            # 如果 CSV 里根本没有 lifetime 这一列，我们手动创建一列全为 -1
            if 'lifetime' not in df.columns:
                print("提示: 未发现 'lifetime' 列，将默认全设为 -1。")
                df['lifetime'] = -1
            else:
                # 如果有这一列，把里面的 NaN (空值) 填充为 -1
                # 并确保转换为 float 类型
                df['lifetime'] = df['lifetime'].fillna(-1).astype(float)
            
            # =================================================
            
            self.data = df
            print(f"数据加载完成，有效样本数: {len(self.data)}")

        except Exception as e:
            print(f"读取 CSV 失败: {e}")
            raise e

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        返回: (sequence_string, lifetime_value)
        """
        # 获取当前行
        row = self.data.iloc[idx]
        
        # 1. 处理序列
        sequence = str(row['seq'])
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
            
        # 2. 处理 lifetime
        lifetime = float(row['lifetime'])
        
        # 返回元组 (序列, 参数)
        return sequence, lifetime



