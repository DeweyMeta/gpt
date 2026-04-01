# train.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken # OpenAI 的 BPE 分词器
import time
from tqdm import tqdm # 用于显示进度条

from train_gpt2 import GPT, GPT2_CONFIG

# -------------------- 1. 数据集准备 --------------------
class TextDataset(Dataset):
    """
    一个简单的文本数据集，将整个文本文件 tokenize 后，
    切分成多个固定长度的输入-目标序列对。
    """
    def __init__(self, file_path, seq_len, tokenizer):
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        # 读取并编码整个文件
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        # tiktoken 的 gpt2 编码器
        self.tokens = tokenizer.encode(text)
        print(f"Total tokens in dataset: {len(self.tokens)}")

        # 计算可以生成多少个样本
        self.num_samples = (len(self.tokens) - 1) // seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        对于每个样本，返回输入序列 x 和目标序列 y。
        x 是 [idx: idx+seq_len] 的 token
        y 是 [idx+1: idx+seq_len+1] 的 token (即 x 向右偏移一位)
        """
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len
        # 输入的 token 序列
        x = torch.tensor(self.tokens[start_idx:end_idx], dtype=torch.long)
        # 目标 token 序列 (下一个词)
        y = torch.tensor(self.tokens[start_idx+1:end_idx+1], dtype=torch.long)
        return x, y

# -------------------- 2. 训练函数 --------------------
def train_model(model, train_loader, val_loader, optimizer, device, num_epochs, eval_interval=100):
    """完整的训练循环"""
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")

        for batch_idx, (x, y) in progress_bar:
            x, y = x.to(device), y.to(device)

            # 1. 前向传播
            logits, loss = model(x, y)

            # 2. 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 3. 梯度裁剪 (防止梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 4. 参数更新
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)

            # 更新进度条显示
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{avg_loss:.4f}"})

            # 定期评估和生成
            if (batch_idx + 1) % eval_interval == 0:
                evaluate_and_generate(model, val_loader, device, batch_idx+1, epoch+1)

    print("Training completed!")

def evaluate_and_generate(model, val_loader, device, step, epoch):
    """在验证集上评估损失并生成一个示例文本"""
    model.eval()
    with torch.no_grad():
        # 计算验证损失 (只取一个 batch 做快速检查)
        val_batch = next(iter(val_loader))
        x_val, y_val = [t.to(device) for t in val_batch]
        _, val_loss = model(x_val, y_val)
        print(f"\n[Epoch {epoch}, Step {step}] Validation Loss: {val_loss.item():.4f}")

        # 生成文本示例
        context = torch.tensor([[tokenizer.encode("The ")[0]]]).to(device) # 起始词 "The"
        generated = model.generate(context, max_new_tokens=50, temperature=0.8)
        generated_text = tokenizer.decode(generated[0].tolist())
        print(f"Generated sample: {generated_text}\n")
    model.train()

# -------------------- 3. 主程序入口 --------------------
if __name__ == "__main__":
    # --- 配置参数 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    BATCH_SIZE = 4          # 根据你的 GPU 内存调整
    NUM_EPOCHS = 50          # 可以先用 1 个 epoch 测试
    LEARNING_RATE = 3e-4
    EVAL_INTERVAL = 50      # 每 50 个 batch 评估一次
    DATA_FILE = "input.txt" # 这里放你下载的 TinyShakespeare 文件

    # 模型配置 (使用你之前定义的 GPT2_CONFIG)
    # 但为了训练，需要确保 vocab_size 与 tokenizer 匹配
    tokenizer = tiktoken.get_encoding('gpt2')
    GPT2_CONFIG["vocab_size"] = tokenizer.n_vocab # 设置为 50257

    # --- 准备数据 ---
    print("Loading dataset...")
    dataset = TextDataset(DATA_FILE, GPT2_CONFIG["seq_len"], tokenizer)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 为了简单起见，这里复用训练集作为验证集，实际应用应划分
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    # --- 初始化模型和优化器 ---
    print("Initializing model...")
    model = GPT(GPT2_CONFIG)

    # 可选：加载预训练权重 (如果你有的话)
    # checkpoint = torch.load("your_checkpoint.pth")
    # model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # --- 开始训练 ---
    print("Starting training...")
    train_model(model, train_loader, val_loader, optimizer, DEVICE, NUM_EPOCHS, EVAL_INTERVAL)

    # --- 保存模型 ---
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': GPT2_CONFIG,
    }, "gpt2_trained.pth")
    print("Model saved to gpt2_trained.pth")