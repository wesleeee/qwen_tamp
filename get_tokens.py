from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
import numpy as np
from get_test_inputs import get_test_inputs
def get_input_tokens_per_block(model_path, image_path, prompt, device="cuda:0"):
    input_tokens = []

    # 加载模型和处理器
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="cuda:0"
    ).to(device)
    processor = AutoProcessor.from_pretrained(model.config._name_or_path)

    # 模拟消息格式，用于模型输入
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
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
    inputs = inputs.to(device)

    # 定义一个hook函数来捕获每个block的输入token值

    def save_input_tokens(module, input, output):
        if input:
            # print(f"Module: {module}")
            # print(f"Input shape: {input[0].shape}")  # 打印输入的张量形状

            input_tokens.append(input[0].detach().to(torch.float).cpu().numpy()[0])  # 仅添加张量部分
        else:
            print(f"Empty input for {module}")
            # print(output)

    # 为模型中的每个transformer block添加hook
    hooks = []

    for block_idx, block in enumerate(model.language_model.layers):
        attention_hook = block.register_forward_hook(save_input_tokens)
        hooks.append(attention_hook)

    model.eval()

    # 打印inputs的详细信息
    # print(f"Inputs: {inputs}")

    with torch.no_grad():
        model_output = model.generate(**inputs, max_new_tokens=128)

    # 清理hook
    for hook in hooks:
        hook.remove()

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], model_output)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    import numpy as np
    import math

    num_groups = 36
    input_m = []

    # ① 先找统一长度 maxi
    maxi = max(len(toks) for toks in input_tokens)

    # ② 逐组处理：第 i 组拿索引 i, i+36, i+72...
    for i in range(num_groups):
        group_sum = None  # 累加器
        count = 0  # 实际累加了多少条（最后一组可能数量不齐）

        for idx in range(i, len(input_tokens), num_groups):
            toks = input_tokens[idx]
            cur_len = len(toks)

            # —— 对齐到 maxi —— #
            if cur_len < maxi:  # 均值填充
                cur_mean = np.mean(toks)
                if toks.ndim == 2:  # 2-D 情况
                    padded = np.full((maxi, toks.shape[1]), cur_mean)
                    padded[:cur_len] = toks
                else:  # 1-D 情况
                    padded = np.full(maxi, cur_mean)
                    padded[:cur_len] = toks
            else:  # 截断
                padded = toks[:maxi]

            # —— 累加 —— #
            if group_sum is None:
                group_sum = padded.astype(np.float64)
            else:
                group_sum += padded
            count += 1

        # ③ 求平均并保存
        input_m.append(group_sum / count)

    # for i in range(len(input_tokens) // 2):
    #     a = (input_tokens[i * 2][:mini] + input_tokens[i * 2 + 1][:mini] ) / 2
    #     input_m.append(a)

    return input_m

