from tqdm import tqdm
import torch.nn as nn
import torch
from accelerate import Accelerator
from tqdm import tqdm

def train_one_epoch_mask_predict(model, dataloader, optimizer, accelerator, epoch_index):
    model.train()
    
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # 【1】控制进度条：只在主进程（Rank 0）显示
    # 如果不加 disable，4张卡会打印4个进度条，刷屏很难看
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch_index}", disable=not accelerator.is_local_main_process)
    
    for step, batch in enumerate(pbar): # enumerate for step counting
        with accelerator.accumulate(model): # 效果：自动帮你做梯度累积。 当 step % accelerator.gradient_accumulation_steps == 0 时，才会真正执行 optimizer.step() 和 optimizer.zero_grad()，其他时候只是计算 loss 和 backward，但不更新参数。
            input_ids, labels, attention_mask, lifetime = batch

            input_ids = input_ids.to(accelerator.device)
            labels = labels.to(accelerator.device)
            attention_mask = attention_mask.to(accelerator.device)

            # 这里的 zero_grad 通常不需要手动写，accelerator.accumulate 会帮你处理
            # 但写了也不报错，为了保险可以先保留，或者移到 step 之后
            # optimizer.zero_grad() 
            
            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs["logits"]
            
            loss = criterion(logits.permute(0, 2, 1), labels)
            
            # 反向传播
            accelerator.backward(loss)
            
            # 参数更新
            optimizer.step()
            optimizer.zero_grad() # 建议放在 step 之后
            
            # 【优化】只记录累积步那一刻的 Loss，避免打印波动太大
            # sync_gradients 为 True 表示当前步确实进行了参数更新
            if accelerator.sync_gradients:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        total_loss += loss.item()
            
        
    # 计算当前 GPU 上的平均 loss
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate_mask_predict(model, dataloader, accelerator):
    """
    验证函数（严谨版）
    采用流式计算+加权平均，确保分布式训练下数学指标的准确性
    """
    model.eval()

    local_loss_sum = 0.0 # 为什么要分local与global？因为多卡训练时，每张卡上数据量可能不一样，所以每个卡都有自己的local统计
    local_token_count = 0
    local_correct_count = 0

    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')  # sum for total loss
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process)
        for batch in pbar:
            input_ids, labels, attention_mask, *others = batch
            
            input_ids = input_ids.to(accelerator.device)
            labels = labels.to(accelerator.device)
            attention_mask = attention_mask.to(accelerator.device)

            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs["logits"]

            # 计算 Loss
            local_loss = criterion(logits.permute(0, 2, 1), labels)
            
            # 计算有效 Token 数量（非 -100 的位置）
            num_valid_tokens = (labels != -100).sum().item()
            
            if num_valid_tokens > 0:
                # 累加loss
                local_loss_sum += local_loss # 累积加权 Loss

                # 累加token数
                local_token_count += num_valid_tokens

                # 计算正确预测的Token数量
                predictions = logits.argmax(dim=-1)
                correct = (predictions == labels) & (labels != -100)
                local_correct_count += correct.sum().item()
    
    stats_tensor = torch.tensor(
        [local_loss_sum, local_token_count, local_correct_count], 
        dtype=torch.float64, # 使用双精度，防止累加过多产生浮点误差
        device=accelerator.device
    )
    # 使用 reduce(sum) 将所有显卡的数据直接相加
    # 结果 = [GPU0_loss+GPU1_loss, GPU0_tokens+GPU1_tokens, ...]
    global_stats = accelerator.reduce(stats_tensor, reduction="sum") #accelerator.reduce 会自动处理分布式环境下的通信，确保所有 GPU 上的数据正确汇总到一起。
    # 解包全局数据
    global_loss_sum = global_stats[0].item()
    global_token_cnt = global_stats[1].item()
    global_correct_cnt = global_stats[2].item()
    if global_token_cnt > 0:
        # 全局平均 Loss = 全局总 Loss / 全局总 Token 数
        avg_loss = global_loss_sum / global_token_cnt
        # 全局准确率
        accuracy = (global_correct_cnt / global_token_cnt) * 100
    else:
        # 防止除以零 (例如验证集可能太小或者全是 Pad)
        avg_loss = 0.0
        accuracy = 0.0
    
    return avg_loss, accuracy