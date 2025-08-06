from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

def get_token_sources(model_path, image_path, prompt, device="cuda:0"):
    """
    获取每个transformer block的输入token的来源：哪些token来自于文本，哪些来自于图像。

    参数:
        model_path (str): 预训练模型的路径
        image (PIL.Image): 输入图像
        prompt (str): 输入文本提示
        device (str): 设备类型, 默认使用"cpu"

    返回:
        dict: 每个block对应的输入token来源，标记哪些来自文本，哪些来自图像
    """
    # 初始化字典来保存每个block的输入token来源
    # token_sources_dict = {}

    # 加载模型和处理器
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path).to(device)
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
    # print(image_inputs)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)


    image_len=[]
    def save_input_tokens_and_sources(module, input, output):
        # input[0]是输入的token表示
        # input_tokens = input[0].cpu().detach().numpy()  # (batch_size, seq_len, embedding_dim)


        output_tokens = output.detach().to(torch.float).cpu().numpy()
        image_len.append(len(output_tokens))


    # 为模型中的每个transformer block添加hook
    hook=model.visual.register_forward_hook(save_input_tokens_and_sources)
    # for block_idx, block in enumerate(model.encoder.block):
    #     # 注册hook捕获每个block的输入token和来源
    #     attention_hook = block.attention.self.register_forward_hook(save_input_tokens_and_sources)
    #     hooks.append(attention_hook)

    # 执行推理
    model.to(device)
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
    # for hook in hooks:
    #     hook.remove()
    hook.remove()
    print(image_len[0])

    return image_len[0]

#
# # 示例调用
# model_path =  "/home/chuziyuan/zju/TAMP/checkpoints/VLM-R1-Qwen2.5VL-3B-OVD-0321"
# image_path = "/home/chuziyuan/zju/qwen_tamp/test_image/蓝桥杯.jpg"
# prompt = "Describe the scene in the image."
#
# # 获取每个block的token来源（文本或图像）
# token_sources = get_token_sources(model_path, image_path, prompt, device="cuda:0")

# 打印每个block的token来源
# for block, sources in token_sources.items():
    # print(f"Block {block}: Token sources: {sources}")


# return as follows:
# Block 0: Token sources: ['text', 'text', 'text', 'image', 'image', 'image']
# Block 1: Token sources: ['text', 'text', 'text', 'image', 'image', 'image']
# Block 2: Token sources: ['text', 'text', 'text', 'image', 'image', 'image']
# ...
