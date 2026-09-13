from tqdm import tqdm
import torch.nn as nn
import torch
def train_one_epoch_mask_predict(model, dataloader, optimizer, device, epoch_index):
    """
    训练一个 Epoch
    """
    model.train() # 切换到训练模式
    
    total_loss = 0.0
    # 定义 Loss 函数 (忽略 -100)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    # 使用 tqdm 显示进度条
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch_index}")
    
    for batch in pbar:
        # batch 解包：由 ESMCMaskedLM.__call__ 返回的元组
        input_ids, labels, attention_mask, lifetime = batch

        # 1. 清零梯度
        optimizer.zero_grad()
        
        # 2. 前向传播
        outputs = model(input_ids=input_ids,labels=labels,attention_mask=attention_mask);
        
        logits = outputs["logits"]
        # 3. 计算 Loss
        # logits: [B, Seq, Vocab] -> [B, Vocab, Seq]
        loss = criterion(logits.permute(0, 2, 1), labels)
        
        # 4. 反向传播
        loss.backward()
        
        # 5. 参数更新
        optimizer.step()
        
        # 记录统计
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate_mask_predict(model, dataloader, device):
    """
    验证函数
    """
    model.eval() # 切换到评估模式
    
    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    with torch.no_grad(): # 不计算梯度
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids, labels, attention_mask, *others = batch
            
            # 前向传播
            outputs = model(input_ids=input_ids,labels=labels,attention_mask=attention_mask);
            
            logits = outputs["logits"]
            # 计算 Loss
            
            loss = criterion(logits.permute(0, 2, 1), labels)
            total_loss += loss.item()
            
            # --- 计算准确率 (Masked Accuracy) ---
            predictions = torch.argmax(logits, dim=-1)
            
            # 只统计 label 不为 -100 的位置
            mask_positions = (labels != -100)
            correct = (predictions == labels) & mask_positions
            
            total_correct += correct.sum().item()
            total_masked += mask_positions.sum().item()
            
    avg_loss = total_loss / len(dataloader)
    accuracy = (total_correct / total_masked * 100) if total_masked > 0 else 0.0
    
    return avg_loss, accuracy