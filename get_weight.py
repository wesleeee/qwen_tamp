from transformers import Qwen2_5_VLForConditionalGeneration
import torch


def extract_qkv_weights_for_pruning(model_path):
    """
    提取Qwen模型中每个transformer layer中的Q、K、V线性层的权重，并返回一个字典，包含每个layer的Q、K、V线性层的权重。

    参数:
        model_path (str): Qwen模型的路径

    返回:
        dict: 包含每个layer的Q、K、V线性层的权重，键为layer索引，值为Q、K、V线性层的权重
    """
    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, device_map=device)

    # 存储权重的字典
    qkv_weights_dict = {}

    # 遍历所有的transformer layer（即language_model.layers）
    for layer_idx, layer in enumerate(model.language_model.layers):
        # print(f"--- Transformer Layer {layer_idx + 1} ---")
        layer_qkv_weights = {}

        # 获取self-attn部分
        attention = layer.self_attn

        # 提取QKV线性层的权重
        for name, module in attention.named_modules():
            if isinstance(module, torch.nn.Linear):
                # 根据模块名称判断是否是Q、K、V线性层
                if "q_proj" in name:  # Q线性层
                    # print(f"Layer {layer_idx}: Query Linear Layer")
                    layer_qkv_weights["Query Linear Layer"] = module.weight.data.cpu().numpy()
                elif "k_proj" in name:  # K线性层
                    # print(f"Layer {layer_idx}: Key Linear Layer")
                    layer_qkv_weights["Key Linear Layer"] = module.weight.data.cpu().numpy()
                elif "v_proj" in name:  # V线性层
                    # print(f"Layer {layer_idx}: Value Linear Layer")
                    layer_qkv_weights["Value Linear Layer"] = module.weight.data.cpu().numpy()

        # 将每个layer的Q、K、V线性层的权重保存到字典
        if layer_qkv_weights:
            qkv_weights_dict[f"Layer_{layer_idx}"] = layer_qkv_weights

    return qkv_weights_dict


