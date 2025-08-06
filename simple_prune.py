import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch.nn.utils.prune as prune
import gc





# 添加支持多模态输入的剪枝函数
def prune_model_with_image(rate, image_path=None):
    """支持图像输入的剪枝函数"""
    model_id = "/home/chuziyuan/zju/TAMP/checkpoints/VLM-R1-Qwen2.5VL-3B-Math-0305"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- 使用设备: {device} ---")

    print(f"--- 正在加载模型: {model_id} ---")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype="auto",
        device_map=device  # 使用单个设备而不是auto
    )
    processor = AutoProcessor.from_pretrained(model_id)

    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        nonzero_params = sum(torch.count_nonzero(p).item() for p in model.parameters())
        return total_params, nonzero_params

    print("\n--- 步骤 2: 剪枝前分析 ---")
    total_params_before, nonzero_params_before = count_parameters(model)
    print(f"剪枝前 - 总参数: {total_params_before:,}")
    print(f"剪枝前 - 非零参数: {nonzero_params_before:,}")

    # 构建多模态输入（包含图像）
    # if image_path:
    #     messages = [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "image", "image": image_path},
    #                 {"type": "text", "text": "tell me about the picture"}
    #             ]
    #         }
    #     ]
    # else:
    #     messages = [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": "你好，请介绍一下你自己，并解释一下什么是计算机视觉。"}
    #             ]
    #         }
    #     ]

    # 预处理
    # text = processor.apply_chat_template(
    #     messages, tokenize=False, add_generation_prompt=True
    # )
    # image_inputs, video_inputs = process_vision_info(messages)
    # inputs = processor(
    #     text=[text],
    #     images=image_inputs,
    #     videos=video_inputs,
    #     padding=True,
    #     return_tensors="pt"
    # ).to(device)

    # 剪枝前生成
    # output_before = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    # generated_ids_trimmed = [
    #     out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_before)
    # ]
    # text_before = processor.batch_decode(
    #     generated_ids_trimmed,
    #     skip_special_tokens=True,
    #     clean_up_tokenization_spaces=False
    # )[0]

    # 执行剪枝（仅剪枝QKV层的权重）
    parameters_to_prune = []
    print(f"\n--- 步骤 3: 执行{rate * 100}%剪枝操作 ---")

    for layer_idx, layer in enumerate(model.language_model.layers):
        print(f"正在剪枝 layer {layer_idx + 1} 的 QKV 线性层...")

        attention = layer.self_attn

        for name, module in attention.named_modules():
            if isinstance(module, torch.nn.Linear):
                if "q_proj" in name or "k_proj" in name or "v_proj" in name:
                    parameters_to_prune.append((module, 'weight'))

    print(f"开始对{len(parameters_to_prune)}个QKV线性层进行{rate * 100}%的L1非结构化剪枝...")
    for i, (module, name) in enumerate(parameters_to_prune):
        if i % 10 == 0:  # 每10层打印一次进度
            print(f"正在剪枝第 {i + 1}/{len(parameters_to_prune)} 层...")
        prune.l1_unstructured(module, name, amount=rate)

    # 固化剪枝
    print("\n--- 步骤 4: 固化剪枝 ---")
    for module, name in parameters_to_prune:
        prune.remove(module, name)

    # 剪枝后分析
    print("\n--- 步骤 5: 剪枝后分析 ---")
    gc.collect()
    torch.cuda.empty_cache()

    total_params_after, nonzero_params_after = count_parameters(model)
    print(f"剪枝后 - 总参数: {total_params_after:,}")
    print(f"剪枝后 - 非零参数: {nonzero_params_after:,}")
    reduction = (total_params_before - nonzero_params_after) / total_params_before
    print(f"参数有效减少比例: {reduction:.2%}")

    # # 剪枝后生成
    # inputs = processor(
    #     text=[text],
    #     images=image_inputs,
    #     videos=video_inputs,
    #     padding=True,
    #     return_tensors="pt"
    # ).to(device)

    # output_after = model.generate(**inputs, max_new_tokens=500, do_sample=False)
    # generated_ids_trimmed_after = [
    #     out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_after)
    # ]
    # text_after = processor.batch_decode(
    #     generated_ids_trimmed_after,
    #     skip_special_tokens=True,
    #     clean_up_tokenization_spaces=False
    # )[0]

    # print("\n\n========== 最终对比 ==========")
    # print(f"剪枝前总参数: {total_params_before:,}")
    # print(f"剪枝后有效参数: {nonzero_params_after:,}")
    # print("-" * 20)
    # print("【剪枝前回答】:")
    # print(text_before)
    # print("-" * 20)
    # print("【剪枝后回答 (无微调)】:")
    # print(text_after)

    save_path = "/home/chuziyuan/zju/qwen_tamp/simple_pruned_qwen_model_math"
    model.save_pretrained(save_path, safe_serialization=True)  # 使用 safetensors 格式
    processor.save_pretrained(save_path)
    print(f"模型已以 safetensors 格式保存到: {save_path}")

    return model, processor


if __name__ == "__main__":

    
    # 多模态剪枝测试（如果有图像文件的话）
    print("\n\n=== 多模态剪枝测试 ===")
    prune_model_with_image(0.75, image_path="/home/chuziyuan/zju/qwen_tamp/test_image/蓝桥杯.jpg")
