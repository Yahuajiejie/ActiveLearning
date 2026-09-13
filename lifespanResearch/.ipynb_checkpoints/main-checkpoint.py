import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import AutoModelForMaskedLM, AutoTokenizer

# --- 导入自定义模块 ---
# 确保 module 文件夹在 Python 搜索路径中
from module.dataset import ProteinDataset,ESMCDataCollator
from module.train import train_one_epoch, evaluate
from module.model import ESMCForMaskedLM
# from module.filter import filter_sequences # 如果你有写过滤逻辑，可以在这里导入

def load_model_and_tokenizer(model_name="wukong/esmc-600m"):
    """
    加载 ESMC 模型和 Tokenizer。
    请根据你实际使用的模型名称或本地路径修改 model_name。
    """
    print(f"正在加载模型: {model_name} ...")
    try:
        # 这里假设使用的是 Hugging Face 格式的 ESMC
        # 如果是 EvolutionScale 原生库，请替换为对应的 esmc.load_model()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        return model, tokenizer
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请检查 model_name 是否正确，或者网络连接是否正常。")
        raise e

def main():
    # ================= 配置参数 =================
    CSV_PATH = "data.csv"          # 数据文件路径
    MODEL_NAME = "EvolutionaryScale/esmc-300m" # 替换为你具体使用的模型名称/路径
    SAVE_PATH = "best_esmc_finetuned.pt" # 模型保存路径
    
    BATCH_SIZE = 4                 # 显存够大可以调大 (e.g., 8, 16)
    EPOCHS = 10                    # 训练轮数 (1000条数据很少，很快就跑完了)
    LEARNING_RATE = 5e-5           # 学习率
    MAX_LENGTH = 1024              # 序列最大长度
    MASK_RATIO = 0.15              # 掩码比例
    # ===========================================

    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. 加载模型和 Tokenizer
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    model.to(device)

    # 3. 准备数据
    # 实例化 Dataset (读取 CSV)
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"找不到数据文件: {CSV_PATH}，请确保文件在正确位置。")
    
    full_dataset = ProteinDataset(CSV_PATH, max_length=MAX_LENGTH)
    
    # 简单的过滤示例 (如果你在 filter.py 里写了逻辑，可以在这里调用)
    # full_dataset.sequences = filter_sequences(full_dataset.sequences) 
    
    print(f"数据集总量: {len(full_dataset)} 条序列")

    # 划分训练集 (90%) 和验证集 (10%)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 4. 初始化 Data Collator (数据处理器)
    # 这就是我们之前写的那个类，它负责 Tokenization 和 Masking
    collator = ESMCMaskedLM(tokenizer, mask_ratio=MASK_RATIO, device=device)

    # 5. 创建 DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,          # 训练集必须打乱
        collate_fn=collator    # 关键：挂载处理器
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,         # 验证集不打乱
        collate_fn=collator
    )

    # 6. 设置优化器
    # 过滤掉不需要梯度的参数 (通常不需要，除非你冻结了层)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # 7. 开始训练循环
    best_val_loss = float('inf')
    
    print("\n" + "="*30)
    print("开始训练...")
    print("="*30)

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        # 训练一步
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch+1)
        print(f"训练集 Loss: {train_loss:.4f}")
        
        # 验证一步
        if len(val_dataset) > 0:
            val_loss, val_acc = evaluate(model, val_loader, device)
            print(f"验证集 Loss: {val_loss:.4f} | 掩码准确率 (Masked Acc): {val_acc:.2f}%")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), SAVE_PATH)
                print(f"✨ 发现新最佳模型，已保存至: {SAVE_PATH}")
        else:
            print("警告: 验证集为空，跳过验证。")

    print("\n训练结束！")

if __name__ == "__main__":
    main()