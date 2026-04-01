# 1. 导包
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
# 2. 配置参数：总词汇量、单句长度、embedding维度、头的数量、层数、dropout
GPT2_CONFIG = {
    "vocab_size" : 256,
    "seq_len" : 128,
    "embedding_size" : 1024,
    "n_heads" : 8,
    "n_layers" : 6,
    "dropout" : 0.1,
}
# 3. 多头自注意力层
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        
        super().__init__()        #!!!!!!!!!忘写了
        self.embedding_size = cfg["embedding_size"]
        self.n_head = cfg["n_heads"]
        self.head_dim = self.embedding_size // self.n_head  #!!!!!!!!忘写了
        self.seq_len = cfg["seq_len"]
        
        self.WQKV = nn.Linear(self.embedding_size, self.embedding_size * 3)
        self.fc = nn.Linear(self.embedding_size, self.embedding_size)
        self.attn_drop = nn.Dropout(cfg["dropout"])
        self.resi_drop = nn.Dropout(cfg["dropout"])
        
        #! mask写在这里（定义内容）
        mask = torch.tril(torch.ones(self.embedding_size, self.embedding_size))
        #! 设置该参数不更新
        self.register_buffer("mask", mask.view(1, 1, self.seq_len, self.seq_len), persistent=False)
    """
        embedding_size
        n_head
        seq_len
        
        计算QKV层
        全连接层
        
        dropout层（attention-dropout，residual-dropout）
        
        mask
    """
    
    def forward(self, x):       #* x是输入矩阵
        
        b, s, e = x.size()      ## batch_size, seq_len, embedding_dim
        
        qkv = self.WQKV(x)      #? 这里维度不对？
        #! 对的，没毛病
        q, k, v = qkv.split(e, dim = 2)    #? 感觉split里少了点啥
        #! 要把第三维加进去
        
        #! 其实到现在我也不理解这三行啥意思
        q = q.view(b, s, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(b, s, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.n_head, self.head_dim).transpose(1, 2)
        
        #? 下面不知道怎么写了，因为q, k, v都是三维矩阵，不知道怎么把三维矩阵和mask加起来
        # mask = torch.tril(torch.ones(self.embedding_size, self.embedding_size) == 0, float("-inf"))
        #? 下面又忘了，不知道mask矩阵和谁加，只知道处理QKV
        # attention = F.softmax(q, k.transpose(0, 1)) / sqrt(self.embedding_size)
        # attention = attention @ v           #? 我的天啊这对吗
        
        #! 焯！这都啥
        att = (q @ k.transpose(-2, -1)) / sqrt(self.head_dim)
        att = att.mask_fill(self.mask[:, :, :s, :s] == 0, float("-inf"))
        att = F.softmax(att, dim = -1)
        att = self.attn_drop(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, s, e)
        
        y = self.fc(y)
        y = self.resi_drop(y)
        return y
        
        # y = self.fc(attention)
        pass
    """
        计算并分割QKV
        
        计算注意力分数，应用mask，softmax，dropout
        
        全连接层
    """
     
# 4. FFN层
class FFN(nn.Module):
    def __init__(self, cfg):
        super.__init__()
        self.embedding_size = cfg["embedding_size"]
        
        self.fc1 = nn.Linear(self.embedding_size, self.embedding_size * 4)
        # self.relu = nn.ReLU()
        #! 上面那句源代码里没给，需要调用的时候直接用
        self.fc2 = nn.Linear(self.embedding_size * 4, self.embedding_size)
        self.dropout = nn.Dropout(cfg["dropout"])
        pass
    """
        embedding_size
        
        全连接层1，激活函数，全连接层2，dropout
    """
    
    def forward(self, x):
        
        x = self.fc1(x)
        # x = self.relu(x)
        #! 用的F.gelu
        x = F.gelu(x, approximate="tanh")
        x = self.fc2(x)
        x = self.dropout(x)
        
        return x            #? 依然感觉怪怪的
        pass
    """
        全连接层1，激活函数，全连接层2，dropout
    """
    
# 5. Transformer Block
class Block(nn.Module):
    def __init__(self, cfg):
        
        super.__init__()
        self.embedding_size = cfg["embedding_size"]
        
        # self.blocks = nn.ModuleList([
            # nn.LayerNorm(),             #? LayerNorm从哪里来？自己定义一下还是直接用？CausalSelfAttention和FFN怎么传进来？
            # 
        # ])
        self.ln1 = nn.LayerNorm(self.embedding_size)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(self.embedding_size)
        self.ffn = FFN(cfg)
        pass
    """
        embedding_size
        
        LayerNorm，CausalSelfAttention，LayerNorm，FFN
    """
    
    def forward(self, x):
        
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2)
        return x
        pass
    """
        LayerNorm，CausalSelfAttention，LayerNorm，FFN
    """
    
# 6. GPT模型
class GPT(nn.Module):
    def __init__(self, cfg):
        
        super().__init__()
        self.vocab_size = cfg["vocab_size"]
        self.seq_len = cfg["seq_len"]
        self.embedding_size = cfg["embedding_size"]
        self.n_head = cfg["n_heads"]
        self.n_layer = cfg["n_layers"]
        self.dropout = cfg["dropout"]
        
        #? token_embedding和position_embedding怎么来的？
        
        #? 同理，blocks和LayerNorm怎么传进来的？
        pass
    """
        vocab_size
        seq_len
        embedding_size
        n_head
        n_layer
        dropout
        
        token_embedding
        position_embedding
        
        blocks
        LayerNorm
        
    """
    
    def forward(self, x):
        pass
    """
        计算token_embedding和position_embedding
        
        依次通过Transformer Block
        
        最后LayerNorm，输出logits
    """
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        pass
    """
        idx: (B, T) 已生成的token ids
        max_new_tokens: 需要生成的token数量
        迭代生成新token，直到达到max_new_tokens
    """
    
if __name__ == "__main__":
    model = GPT(GPT2_CONFIG)
    x = torch.randint(0, GPT2_CONFIG["vocab_size"], (2, 32))
    y = torch.randint(0, GPT2_CONFIG["vocab_size"], (2, 32))
    logits, loss = model(x, y)
    print("logits:", logits.shape)  # (2, 32, vocab_size)
    print("loss:", float(loss))