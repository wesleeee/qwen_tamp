from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import numpy as np

def get_mean_attention_values(model_path, image_path, prompt, device="cuda:0"):
    """
    获取每个transformer层中的multihead attention矩阵（多个head之间的均值）。

    参数:
        model_path (str): 预训练模型的路径
        image_path (str): 输入图像路径
        prompt (str): 输入文本提示
        device (str): 设备类型, 默认使用"cpu"

    返回:
        dict: 每个layer对应的attention得分的均值
    """
    # 初始化列表来保存每个layer的attention均值
    attention_means = []

    # 加载模型和处理器
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="cuda:0"
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_path)

    # 模拟消息格式，用于模型输入
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text":prompt},
            ],
        }
    ]

    # 使用处理器准备文本和图像输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # 定义一个hook函数来捕获attention值
    def save_attention_scores(module, input, output):
        # print(input)
        # 只使用第一个元素作为attention scores
        if output is not None and isinstance(output, tuple) and len(output) > 0:
            # print("output is {}".format(output))
            attention_scores = output[0]  # 使用第一个元素
            # 对所有heads求均值
            mean_attention = attention_scores.mean(dim=0)  # 对所有head求均值


            # 将mean_attention添加到列表中
            attention_means.append(mean_attention.detach().to(torch.float).cpu().numpy())
        else:
            print(f"Layer did not return valid attention scores.")

    # 为每个transformer layer添加hook（每个层只注册一次hook）
    hooks = []
    for layer_idx, layer in enumerate(model.language_model.layers):  # 访问每个解码器层
        # 注册hook捕获multihead attention，确保每层只有一个hook
        attention_hook = layer.self_attn.register_forward_hook(save_attention_scores)
        hooks.append(attention_hook)

    # 执行推理
    model.eval()


    # 输入数据进入模型
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    # 清理hook
    for hook in hooks:
        hook.remove()

    num_groups = 36
    attn_m = []
    maxi = max(len(a) for a in attention_means)
    for i in range(num_groups):
        group_sum = None
        cnt = 0
        for idx in range(i, len(attention_means), num_groups):
            a = attention_means[idx]
            cur_len = len(a)
            if cur_len < maxi:
                cur_mean = np.mean(a)
                if a.ndim == 2:
                    padded = np.full((maxi, a.shape[1]), cur_mean)
                    padded[:cur_len] = a
                else:
                    padded = np.full(maxi, cur_mean)
                    padded[:cur_len] = a
            else:
                padded = a[:maxi]

            if group_sum is None:
                group_sum = padded.astype(np.float64)
            else:
                group_sum += padded
            cnt += 1
        attn_m.append(group_sum / cnt)

    return attn_m



