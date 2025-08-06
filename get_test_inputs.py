#!/usr/bin/env python3
"""
从MATH-V多模态数据集中生成30个测试输入组合
数据路径：/home/chuziyuan/zju/qwen_tamp/math-v/MATH-V/
"""

import json
import os
import random
from typing import List, Tuple


def generate_test_inputs_from_dataset() -> Tuple[List[str], List[str]]:
    """
    从MATH-V数据集中生成30个测试输入组合

    Returns:
        Tuple[List[str], List[str]]: (prompts, images) 两个对应的列表
    """

    # 数据集路径
    jsonl_file = "/home/chuziyuan/zju/qwen_tamp/math-v/MATH-V/data/test.jsonl"
    
    # 设置随机种子确保可重现
    random.seed(42)

    all_samples = []

    try:
        # 读取JSONL文件
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    
                    # 提取问题作为prompt，移除<image1>标记
                    question = data.get("question", "")
                    prompt = question.replace("<image1>", "").strip()
                    
                    # 添加选项到prompt（如果是选择题）
                    options = data.get("options", [])
                    if options:
                        options_text = "\n".join([f"{option}." for option in options])
                        prompt = f"{prompt}\n\nOptions:\n{options_text}"
                    
                    # 获取图片路径
                    image_path = data.get("image", "")
                    if not image_path:
                        # 如果没有image字段，根据id构建路径
                        image_id = data.get("id", "")
                        image_path = f"/home/chuziyuan/zju/qwen_tamp/math-v/MATH-V/images/{image_id}.jpg"
                    else:
                        # 如果image字段是相对路径，转换为绝对路径
                        if not os.path.isabs(image_path):
                            if image_path.startswith("images/"):
                                image_path = f"/home/chuziyuan/zju/qwen_tamp/math-v/MATH-V/{image_path}"
                            else:
                                image_path = f"/home/chuziyuan/zju/qwen_tamp/math-v/MATH-V/images/{os.path.basename(image_path)}"
                    
                    # 检查图片文件是否存在
                    if os.path.exists(image_path):
                        all_samples.append((prompt, image_path))
                    else:
                        print(f"Warning: 图片文件不存在: {image_path}")

    except Exception as e:
        print(f"Error: 无法读取数据文件 {jsonl_file}: {e}")
        return [], []

    if not all_samples:
        print("Error: 没有找到有效的数据样本")
        return [], []

    # 随机选择30个样本
    if len(all_samples) >= 1:
        selected_samples = random.sample(all_samples, 1)
    else:
        # 如果样本不够，就重复采样
        selected_samples = random.choices(all_samples, k=1)

    # 分离prompts和images
    prompts = []
    images = []
    for prompt, image in selected_samples:
        prompts.append(prompt)
        images.append(image)

    return prompts, images


# 直接生成并返回结果
def get_test_inputs() -> Tuple[List[str], List[str]]:
    """
    返回30个测试输入组合 - 动态从MATH-V数据集中提取

    Returns:
        Tuple[List[str], List[str]]: (prompts, images) 两个对应的列表
    """

    # 动态从MATH-V数据集中生成
    return generate_test_inputs_from_dataset()


def print_generated_inputs():
    """打印生成的输入并展示格式"""
    prompts, images = generate_test_inputs_from_dataset()

    print("# 从多模态数据集生成的5个测试输入")
    print("# 基于 MATH-V 数据集\n")

    print("prompts = [")
    for i, prompt in enumerate(prompts):
        # 截断过长的prompt用于显示
        display_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
        display_prompt = display_prompt.replace('\n', '\\n')
        print(f'    "{display_prompt}",  # {i + 1}')
    print("]\n")

    print("images = [")
    for i, image in enumerate(images):
        print(f'    "{image}",  # {i + 1}')
    print("]\n")

    print(f"总共生成了 {len(prompts)} 个输入组合")
    print("prompts[i] 和 images[i] 是对应的")

    # 验证文件是否存在
    print("\n验证文件存在性:")
    existing_count = 0
    for i, image_path in enumerate(images[:5]):  # 只检查前5个
        if os.path.exists(image_path):
            print(f"✅ {i + 1}: {os.path.basename(image_path)} 存在")
            existing_count += 1
        else:
            print(f"❌ {i + 1}: {image_path} 不存在")

    if existing_count > 0:
        print(f"前5个文件中有 {existing_count} 个存在")


if __name__ == "__main__":
    print("=" * 60)
    print("从MATH-V数据集生成测试输入")
    print("=" * 60)

    print_generated_inputs()

    # 保存到文件
    prompts, images = generate_test_inputs_from_dataset()

    with open("real_test_inputs.py", "w", encoding="utf-8") as f:
        f.write("# 从MATH-V数据集生成的测试输入\n")
        f.write("# 基于 /home/chuziyuan/zju/qwen_tamp/math-v/MATH-V/ 目录\n\n")

        f.write("def get_test_inputs():\n")
        f.write("    \"\"\"\n")
        f.write("    返回2个测试输入组合\n")
        f.write("    Returns:\n")
        f.write("        Tuple[List[str], List[str]]: (prompts, images)\n")
        f.write("    \"\"\"\n")
        f.write("    prompts = [\n")
        for prompt in prompts:
            # 转义引号以避免语法错误
            escaped_prompt = prompt.replace('"', '\\"').replace('\n', '\\n')
            f.write(f'        "{escaped_prompt}",\n')
        f.write("    ]\n\n")

        f.write("    images = [\n")
        for image in images:
            f.write(f'        "{image}",\n')
        f.write("    ]\n\n")

        f.write("    return prompts, images\n\n")

        f.write("if __name__ == '__main__':\n")
        f.write("    prompts, images = get_test_inputs()\n")
        f.write("    print(f'生成了 {len(prompts)} 个测试输入')\n")

    print("\n✅ 测试输入已保存到 real_test_inputs.py 文件")
    print("可以通过 from real_test_inputs import get_test_inputs 使用")