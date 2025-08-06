import numpy as np
import torch
import torch.nn.functional as F
from get_weight import extract_qkv_weights_for_pruning
from get_attention import get_mean_attention_values
from get_tokens import get_input_tokens_per_block
from get_token_source import get_token_sources
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import os
from get_test_inputs import get_test_inputs
# 自动选择设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_sparsity(image_len, token_list, device=device):
    """
    token_list: List[torch.Tensor] each shape [N_i, D]
    返回: dict block -> float (稀疏度)
    """
    sparsity_dict = {}
    for b, t_cpu in enumerate(token_list):
        x = t_cpu.to(device)
        V = x[:image_len]
        L = x[image_len:]

        def mean_cos_dist(z: torch.Tensor):
            n = z.size(0)
            if n < 2:
                return torch.tensor(0., device=device)
            zn = F.normalize(z, p=2, dim=1)
            sim = zn @ zn.t()
            idx = torch.triu_indices(n, n, offset=1)
            d = 1 - sim[idx[0], idx[1]]
            return d.mean()

        s_v = mean_cos_dist(V)
        s_l = mean_cos_dist(L)
        if V.size(0) > 0 and L.size(0) > 0:
            Vn = F.normalize(V, p=2, dim=1)
            Ln = F.normalize(L, p=2, dim=1)
            cross = 1 - Vn @ Ln.t()
            s_vl = cross.mean()
        else:
            s_vl = torch.tensor(0., device=device)

        density = (s_v + s_l + s_vl) / 3
        sparsity_dict[b] = (1.0 / density).detach().cpu().item()
    return sparsity_dict



def get_activate_tokens_fast(attention_values, token_list, num_neighbors, sparsity_dict,epsilon=0.0012, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    """
    向量化改写版，减少 Python 循环与 .item() 调用，屏蔽已选 token 并动态选取第二大、第三大…。
    返回: dict block -> np.ndarray shape [1, K, D]
    """
    result = {}
    for b, (attn_cpu, tokens_cpu) in enumerate(zip(attention_values, token_list)):
        # 准备数据
        tokens = tokens_cpu.to(device)                     # [N, D]
        attn_raw = attn_cpu.to(device)
        attn = attn_raw[:, -1] if attn_raw.dim() == 2 else attn_raw  # [N]
        N, D = tokens.shape
        L = min(N, attn.shape[0])
        tokens = tokens[:L]
        attn = attn[:L]
        print("token length is {}".format(L))

        # 1) 归一化并计算距离矩阵 dist[i,j] = 1 - cos(tokens[i], tokens[j])
        normed = F.normalize(tokens, p=2, dim=1)           # [L, D]
        sim = normed @ normed.transpose(0,1)               # [L, L]
        dist = 1.0 - sim
        dist.fill_diagonal_(float('inf'))                  # 排除自身距离

        # 2) 预选每个 token 的最近 num_neighbors 个邻居索引
        neg_dist = -dist                                   # 距离小 → neg_dist 大
        neighbors = torch.topk(neg_dist, num_neighbors, dim=1).indices  # [L, num_neighbors]
        # print(neighbors)

        # 3) 预计算衰减系数矩阵 e_mat[i,j] = exp(-dist[i,j])
        e_mat = torch.exp(-dist)
        print("e_mat is {}".format(e_mat))# [L, L]

        # 4) 贪心选 token
        selected = torch.zeros(L, dtype=torch.bool, device=device)
        cur_attn = attn.clone()                            # [L]
        new_tokens = []
        s = 1.0 / sparsity_dict[b]
        print("max is {}".format(max(cur_attn)))
        # 每个 token 接收来自邻居的传播

        new_attn = []
        for i in range(len(tokens)):
            a_i = cur_attn[i].item()
            # w = 1.0
            for item in neighbors[i]:
                if not selected[item]:  # 只考虑未选中的邻居
                    e = e_mat[i][item].item()
                    a_i += e * cur_attn[item].item()
                    # w += e
            new_attn.append(a_i)

        cur_attn = torch.tensor(new_attn, device=device)


        while True:
            # === 每轮更新 cur_attn ===（参考原始 new_attn 更新机制）

                # 只考虑未被选中的 token
            # attn_source = cur_attn.clone()  # 作为邻居传播源

                # 构造 mask: 不传播来自于已选中的 token
            # mask = (~selected).float()  # [1, L]
            # mask=torch.outer(mask,mask)
            # print(selected)
            # print(mask)
            # neighbor_mask = mask * e_mat  # [L, L]，只让未选中的 token 发出影响
            # print(neighbor_mask)
            # print(neighbor_mask)
            # propagated = (neighbor_mask * attn_source.view(1, -1)).sum(dim=1)  # [L]#需要修改
            # cur_attn = attn + propagated
            # print("cur_attn is {}".format(cur_attn))

            # 屏蔽已选 token
            masked_attn = cur_attn.clone()

            masked_attn[selected] = float('-inf')
            # print("selected: {}".format(selected))
            # print("masked_attn: {}".format(masked_attn))

            # 如果没有可选 token，则退出
            max_val = masked_attn.max()
            if not torch.isfinite(max_val):
                # print("no more")
                break

            # 选出当前最大注意力 token
            idx = int(masked_attn.argmax())
            selected[idx] = True
            new_tokens.append(tokens[idx])

            # 衰减其未被选中的邻居的注意力
            nb = neighbors[idx]  # [num_neighbors]
            unselected_mask = ~selected[nb]
            nb_unselected = nb[unselected_mask]
            if nb_unselected.numel() > 0:
                cur_attn[nb_unselected] -= e_mat[idx, nb_unselected] * cur_attn[idx]
                # cur_attn[nb_unselected] /= (1-torch.clamp(e_mat[idx, nb_unselected], max=0.99))#加权，同时防止除以0

            # 5) 计算停止条件 flag
            # 5.1) 全体 pairs 的平均 e
            tri_i, tri_j = torch.triu_indices(L, L, offset=1)
            Acc = e_mat[tri_i, tri_j].mean()

            # 5.2) 已选 pairs
            sel_idx = selected.nonzero(as_tuple=False).view(-1)
            if sel_idx.numel() > 1:
                si, sj = torch.triu_indices(sel_idx.numel(), sel_idx.numel(), offset=1)
                idx_i = sel_idx[si]; idx_j = sel_idx[sj]
                Acc_prim = e_mat[idx_i, idx_j].mean()
            else:
                Acc_prim = torch.tensor(0.0, device=device)

            # # 5.3) 已选 vs 未选
            # rem_idx = (~selected).nonzero(as_tuple=False).view(-1)
            # if sel_idx.numel() > 0 and rem_idx.numel() > 0:
            #     i_expand = sel_idx.view(-1,1).expand(-1, rem_idx.numel()).reshape(-1)
            #     j_expand = rem_idx.repeat(sel_idx.numel())
            #     Acc_pp = e_mat[i_expand, j_expand].mean()
            # else:
            #     Acc_pp = torch.tensor(0.0, device=device)
            # 5.3) 所有 token vs 已选 token（与原始 numpy 版本一致）
            if sel_idx.numel() > 0:
                all_idx = torch.arange(L, device=device)
                i_expand = all_idx.view(-1, 1).expand(-1, sel_idx.numel()).reshape(-1)
                j_expand = sel_idx.repeat(all_idx.numel())
                Acc_pp = e_mat[i_expand, j_expand].mean()
            else:
                print("error on Acc_pp")
                Acc_pp = torch.tensor(0.0, device=device)

            flag = Acc + Acc_prim - 2 * Acc_pp - 0.1 * np.sqrt(s)+epsilon
            print("the length of new_tokens is {}".format(len(new_tokens)))
            print("flag is {}".format(flag))
            if (flag < 0 and len(new_tokens)>10) or len(new_tokens)>1/3*len(tokens):
                break
            # if flag < 0 and len(new_tokens)>10 or len(new_tokens)==len(tokens):
            #     break
        # 收集结果
        if new_tokens:

            stacked = torch.stack(new_tokens, dim=0).unsqueeze(0)  # [1, K, D]
            result[b] = stacked.cpu().numpy()
            print("layer number is {}".format(b))
            print("the shape of new_tokens is {}".format(result[b].shape))
        else:
            result[b] = np.zeros((1, 0, D), dtype=float)

    return result

# 其余函数保持不变，调用 get_sparsity 与 get_activate_tokens 即可。


def process_one_input(qkv_weights, prompt, image_path, model_path):
    print("-------------------------------------------------------------------------------------")
    print("getting attention...")

    attention_means = get_mean_attention_values(model_path, image_path, prompt, device="cuda:0")
    print("-------------------------------------------------------------------------------------")
    print("getting tokens...")

    input_tokens = get_input_tokens_per_block(model_path, image_path, prompt, device="cuda:0")
    print("-------------------------------------------------------------------------------------")
    print("getting image length...")

    image_len = get_token_sources(model_path, image_path, prompt, device="cuda:0")


    # 转为 torch.Tensor 列表
    token_tensors = [torch.from_numpy(arr).to(device) for arr in input_tokens]
    attn_tensors  = [torch.from_numpy(arr).to(device) for arr in attention_means]

    print("--- Computing sparsity ---")
    sparsity_dict = get_sparsity(image_len, token_tensors, device)
    print("--- Activating tokens ---")
    activate_tokens = get_activate_tokens_fast(attn_tensors, token_tensors, 3, sparsity_dict, device=device)

    new_qkv = {}
    for i in range(36):
        # QKV weight adjust as before
        q_w = np.abs(np.array(qkv_weights[f"Layer_{i}"]["Query Linear Layer"]))
        k_w = np.abs(np.array(qkv_weights[f"Layer_{i}"]["Key Linear Layer"]))
        v_w = np.abs(np.array(qkv_weights[f"Layer_{i}"]["Value Linear Layer"]))
        nt   = activate_tokens[i][0]  # numpy [K,D]
        t = np.linalg.norm(nt, ord=2, axis=0)
        new_qkv[f"Layer_{i}"] = {
            "Query Linear Layer": (q_w * t),
            "Key Linear Layer":   (k_w * t),
            "Value Linear Layer": (v_w * t)
        }
    return new_qkv, sparsity_dict


def process_multiple_input(qkv_weights, prompts, image_paths, model_path):
    all_weights = None
    all_spars = None
    for idx, (p, ip) in enumerate(zip(prompts, image_paths)):
        print(f"Processing input {idx}")
        w, s = process_one_input(qkv_weights, p, ip, model_path)
        if idx == 0:
            all_weights = w
            all_spars = s
        else:
            for k in all_spars:
                all_spars[k] += s[k]
            for layer in all_weights:
                for proj in all_weights[layer]:
                    all_weights[layer][proj] += w[layer][proj]
    # 平均
    n = len(prompts)
    for k in all_spars: all_spars[k] /= n
    for layer in all_weights:
        for proj in all_weights[layer]:
            all_weights[layer][proj] /= n
    return all_weights, all_spars


def pruning(prompts, image_paths, model_path, average_sparsity, pruned_model_path):
    print("--- Extracting weights ---")
    qkv_weights = extract_qkv_weights_for_pruning(model_path)
    new_weights, sparsity = process_multiple_input(qkv_weights, prompts, image_paths, model_path)

    # 计算 mask 并应用到 qkv_weights
    total = sum(sparsity.values())
    k = average_sparsity * 36 / total
    masks = {}
    for i in range(36):
        sp = min(k * sparsity[i], 1)
        key = f"Layer_{i}"
        masks[key] = {}
        for proj in ["Query Linear Layer","Key Linear Layer","Value Linear Layer"]:
            w = np.array(new_weights[key][proj])
            flat = w.flatten()
            cut = int(flat.size * sp)
            idxs = np.argsort(flat)[:cut]
            m = np.ones_like(flat)
            m[idxs] = 0
            masks[key][proj] = m.reshape(w.shape)
            qkv_weights[key][proj] *= masks[key][proj]

    # 写回模型
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    for li, layer in enumerate(model.language_model.layers):
        key = f"Layer_{li}"
        if key not in qkv_weights: continue
        with torch.no_grad():
            layer.self_attn.q_proj.weight.data = torch.from_numpy(qkv_weights[key]["Query Linear Layer"]).to(device)
            layer.self_attn.k_proj.weight.data = torch.from_numpy(qkv_weights[key]["Key Linear Layer"]).to(device)
            layer.self_attn.v_proj.weight.data = torch.from_numpy(qkv_weights[key]["Value Linear Layer"]).to(device)
        print(f"Injected Layer {li}")
    os.makedirs(pruned_model_path, exist_ok=True)
    model.save_pretrained(pruned_model_path, safe_serialization=True)
    processor.save_pretrained(pruned_model_path)
    print(f"Model saved to {pruned_model_path}")

if __name__ == "__main__":
    print("start pruning on math ...")
    prompts,image_paths=get_test_inputs()

    model_path = "/home/chuziyuan/zju/TAMP/checkpoints/VLM-R1-Qwen2.5VL-3B-Math-0305"
    pruned_model_path = "/home/chuziyuan/zju/qwen_tamp/my_pruned_qwen_model_math"
    pruning(prompts, image_paths, model_path, 0.90, pruned_model_path)
