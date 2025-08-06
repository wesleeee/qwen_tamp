import torch
import numpy as np
from typing import List
import torch.nn.functional as F

# 👇请将你自己的 fast 函数粘贴进来
from my_prune_on_gpu import get_activate_tokens_fast  # 替换为你的实际模块路径

def generate_test_data(L=100, D=256, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    tokens = torch.randn(L, D)                # 随机 token 向量
    attention = torch.rand(L, 1)              # 单列注意力值
    return [attention], [tokens]

def test_get_activate_tokens_fast():
    attention_values, token_list = generate_test_data(L=100, D=256)
    num_neighbors = 3
    sparsity_dict = {0: 5}

    print("🔍 Running get_activate_tokens_fast on 100 tokens...")

    results = get_activate_tokens_fast(
        attention_values,
        token_list,
        num_neighbors,
        sparsity_dict,
        epsilon=-0.1  # 可调
    )

    for block, val in results.items():
        shape = val.shape  # [1, K, D]
        num_selected = shape[1]
        print(f"✅ Block {block}: selected {num_selected} tokens out of 100.")
        if num_selected <= 12:
            print("⚠️  Warning: Selected too few tokens — possible early exit!")

if __name__ == "__main__":
    test_get_activate_tokens_fast()
