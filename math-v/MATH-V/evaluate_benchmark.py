import json
import os
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import re
from tqdm import tqdm

def load_model():
    """加载模型和设置参数，与qwen.py中的流程一致"""
    MODEL_PATH = "/home/chuziyuan/zju/qwen_tamp/my_pruned_qwen_model_math"
    
    llm = LLM(
        model=MODEL_PATH,
        limit_mm_per_prompt={"image": 10, "video": 10},
    )
    
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=4096,
        stop_token_ids=[],
    )
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    return llm, sampling_params, processor

def load_test_data(jsonl_path):
    """加载测试数据"""
    test_data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line.strip()))
    return test_data

def create_message_for_question(question_data, image_path):
    """为每个问题创建消息格式，支持选择题和填空题，要求使用boxed格式输出"""
    question = question_data["question"]
    options = question_data["options"]
    
    # 判断是选择题还是填空题
    is_multiple_choice = len(options) > 0
    
    if is_multiple_choice:
        # 选择题处理
        options_text = "\n".join([f"{option}." for option in options])
        full_question = f"{question}\n\nOptions:\n{options_text}\n\nPlease select the correct answer from the given options (A, B, C, D, or E). Please format your answer as \\boxed{{X}} where X is the letter of your chosen option."
        system_message = "You are a helpful assistant. Please answer the multiple choice question by selecting one of the given options. Always format your final answer as \\boxed{letter}. You shouldn't give any other response"
    else:
        # 填空题处理
        full_question = f"{question}\n\nPlease provide a direct and concise answer to this question. Please format your final answer as \\boxed{{answer}} where answer is your only response."
        system_message = "You are a helpful assistant. Please provide a direct and accurate answer to the question. Always format your answer as \\boxed{answer}. You shouldn't give any other response"
    
    messages = [
        {"role": "system", "content": system_message},
        {
            "role": "user", 
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    "min_pixels": 224 * 224,
                    "max_pixels": 1280 * 28 * 28,
                },
                {"type": "text", "text": full_question},
            ],
        },
    ]
    
    return messages

def generate_answer(llm, processor, sampling_params, messages):
    """生成答案，使用与qwen.py相同的流程"""
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    
    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }
    
    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text
    
    return generated_text

def extract_answer(generated_text, question_data):
    """从生成的文本中提取答案，优先解析\\boxed{}格式"""
    
    # 首先尝试提取 \boxed{} 格式的答案
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    boxed_match = re.search(boxed_pattern, generated_text, re.IGNORECASE)
    
    if boxed_match:
        boxed_answer = boxed_match.group(1).strip()
        
        # 对于选择题，确保提取的是单个字母
        options = question_data["options"]
        is_multiple_choice = len(options) > 0
        
        if is_multiple_choice:
            # 从boxed答案中提取选项字母
            letter_match = re.search(r'([A-E])', boxed_answer, re.IGNORECASE)
            if letter_match:
                return letter_match.group(1).upper()
            # 如果boxed中没有找到字母，返回boxed的内容（可能是完整选项）
            return boxed_answer.upper()
        else:
            # 填空题：返回boxed中的内容
            return boxed_answer
    
    # 如果没有找到boxed格式，使用备用方案
    options = question_data["options"]
    is_multiple_choice = len(options) > 0
    
    if is_multiple_choice:
        # 选择题：查找选项字母
        patterns = [
            r'[答案是答案为answer is]\s*([A-E])',
            r'选择\s*([A-E])',
            r'选项\s*([A-E])',
            r'([A-E])\s*[是为]正确',
            r'正确答案[是为]\s*([A-E])',
            r'\b([A-E])\b',  # 单独的字母
        ]
        
        for pattern in patterns:
            match = re.search(pattern, generated_text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # 最后尝试：返回第一个出现的选项字母
        for char in generated_text.upper():
            if char in ['A', 'B', 'C', 'D', 'E']:
                return char
        
        return None
    
    else:
        # 填空题：提取具体答案内容
        cleaned_text = generated_text.strip()
        
        # 移除常见的答案前缀
        prefixes_to_remove = [
            r'^(答案是|答案为|答案：|答案:|Answer is|Answer:|The answer is|The answer:|答案)',
            r'^(结果是|结果为|结果：|结果:|Result is|Result:|The result is|The result:)',
            r'^(总共有|总共是|一共有|一共是)',
            r'^(有|是)',
        ]
        
        for prefix_pattern in prefixes_to_remove:
            cleaned_text = re.sub(prefix_pattern, '', cleaned_text, flags=re.IGNORECASE).strip()
        
        # 移除常见的后缀
        suffixes_to_remove = [
            r'(个|点|分|次|张|只|条|项|种|类)$',
            r'(points?|items?|pieces?)\.?$',
        ]
        
        for suffix_pattern in suffixes_to_remove:
            cleaned_text = re.sub(suffix_pattern, '', cleaned_text, flags=re.IGNORECASE).strip()
        
        # 尝试提取数字
        number_match = re.search(r'\b(\d+(?:\.\d+)?)\b', cleaned_text)
        if number_match:
            return number_match.group(1)

        # 返回清理后的文本
        if cleaned_text:
            first_sentence = re.split(r'[。！？\.\!\?]', cleaned_text)[0]
            return first_sentence[:50].strip()
        
        # 如果清理后为空，返回原始文本的前50个字符
        return generated_text[:50].strip()

def compare_answers(predicted, correct, question_data):
    """比较预测答案和正确答案，支持灵活比较"""
    if predicted is None:
        return False
    
    options = question_data["options"]
    is_multiple_choice = len(options) > 0
    
    if is_multiple_choice:
        # 选择题：直接比较选项字母
        return predicted.upper() == correct.upper()
    else:
        # 填空题：更灵活的比较
        predicted_str = str(predicted).strip().lower()
        correct_str = str(correct).strip().lower()
        
        # 完全匹配
        if predicted_str == correct_str:
            return True
        
        # 移除标点符号再比较
        import string
        predicted_clean = predicted_str.translate(str.maketrans('', '', string.punctuation))
        correct_clean = correct_str.translate(str.maketrans('', '', string.punctuation))
        
        if predicted_clean == correct_clean:
            return True
        
        # 数字比较（考虑小数点差异）
        try:
            predicted_num = float(predicted_str)
            correct_num = float(correct_str)
            # 允许很小的浮点数误差
            return abs(predicted_num - correct_num) < 1e-6
        except (ValueError, TypeError):
            pass
        
        # 如果都是数字，去除前导零比较
        if predicted_str.isdigit() and correct_str.isdigit():
            return int(predicted_str) == int(correct_str)
        
        return False

def evaluate_benchmark():
    """主评测函数"""
    print("正在加载模型...")
    llm, sampling_params, processor = load_model()
    print("模型加载完成！")
    
    # 加载测试数据
    print("正在加载测试数据...")
    test_data = load_test_data("/home/chuziyuan/zju/qwen_tamp/math-v/MATH-V/data/test.jsonl")
    print(f"加载了 {len(test_data)} 条测试数据")
    
    correct_count = 0
    total_count = 0
    results = []
    
    for i, question_data in enumerate(tqdm(test_data, desc="评测进行中")):
        try:
            # 构建图片路径
            image_path = f"/home/chuziyuan/zju/qwen_tamp/math-v/MATH-V/images/{question_data['id']}.jpg"
            
            # 检查图片是否存在
            if not os.path.exists(image_path):
                print(f"警告: 图片 {image_path} 不存在，跳过问题 {question_data['id']}")
                continue
            
            # 创建消息
            messages = create_message_for_question(question_data, image_path)
            
            # 生成答案
            generated_text = generate_answer(llm, processor, sampling_params, messages)
            print(generated_text)
            
            # 提取答案
            predicted_answer = extract_answer(generated_text, question_data)
            correct_answer = question_data["answer"]
            
            # 检查是否使用了boxed格式
            has_boxed_format = bool(re.search(r'\\boxed\{([^}]+)\}', generated_text, re.IGNORECASE))
            
            # 判断是否正确（支持灵活比较）
            is_correct = compare_answers(predicted_answer, correct_answer, question_data)
            if is_correct:
                correct_count += 1
            
            total_count += 1
            
            # 保存结果
            result = {
                "id": question_data["id"],
                "question": question_data["question"],
                "question_type": "multiple_choice" if len(question_data["options"]) > 0 else "fill_in_blank",
                "options": question_data["options"],
                "correct_answer": correct_answer,
                "predicted_answer": predicted_answer,
                "generated_text": generated_text,
                "has_boxed_format": has_boxed_format,
                "is_correct": is_correct
            }
            results.append(result)
            
            # 每10个问题打印一次进度
            if (i + 1) % 10 == 0:
                current_accuracy = correct_count / total_count
                print(f"已完成 {total_count} 个问题，当前准确率: {current_accuracy:.4f}")
        
        except Exception as e:
            print(f"处理问题 {question_data['id']} 时出错: {str(e)}")
            continue
    
    # 计算最终准确率和分类统计
    final_accuracy = correct_count / total_count if total_count > 0 else 0
    
    # 分类统计
    mc_correct = sum(1 for r in results if r["question_type"] == "multiple_choice" and r["is_correct"])
    mc_total = sum(1 for r in results if r["question_type"] == "multiple_choice")
    fib_correct = sum(1 for r in results if r["question_type"] == "fill_in_blank" and r["is_correct"])
    fib_total = sum(1 for r in results if r["question_type"] == "fill_in_blank")
    
    # Boxed格式统计
    boxed_total = sum(1 for r in results if r["has_boxed_format"])
    boxed_correct = sum(1 for r in results if r["has_boxed_format"] and r["is_correct"])
    non_boxed_total = total_count - boxed_total
    non_boxed_correct = correct_count - boxed_correct
    
    print("\n" + "="*60)
    print("评测完成！")
    print(f"总问题数: {total_count}")
    print(f"正确回答数: {correct_count}")
    print(f"最终准确率: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print("-" * 60)
    print("分类统计：")
    if mc_total > 0:
        mc_acc = mc_correct / mc_total
        print(f"选择题 (Multiple Choice): {mc_correct}/{mc_total} = {mc_acc:.4f} ({mc_acc*100:.2f}%)")
    if fib_total > 0:
        fib_acc = fib_correct / fib_total
        print(f"填空题 (Fill in Blank): {fib_correct}/{fib_total} = {fib_acc:.4f} ({fib_acc*100:.2f}%)")
    print("-" * 60)
    print("格式统计：")
    if boxed_total > 0:
        boxed_acc = boxed_correct / boxed_total
        print(f"使用\\boxed格式: {boxed_correct}/{boxed_total} = {boxed_acc:.4f} ({boxed_acc*100:.2f}%)")
    if non_boxed_total > 0:
        non_boxed_acc = non_boxed_correct / non_boxed_total
        print(f"未使用\\boxed格式: {non_boxed_correct}/{non_boxed_total} = {non_boxed_acc:.4f} ({non_boxed_acc*100:.2f}%)")
    print(f"\\boxed格式使用率: {boxed_total}/{total_count} = {boxed_total/total_count*100:.1f}%" if total_count > 0 else "0%")
    print("="*60)
    
    # 保存详细结果到文件
    with open("/home/chuziyuan/zju/qwen_tamp/math-v/MATH-V/outputs/qwen-vl-my/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_questions": total_count,
                "correct_answers": correct_count,
                "accuracy": final_accuracy,
                "multiple_choice": {
                    "total": mc_total,
                    "correct": mc_correct,
                    "accuracy": mc_correct / mc_total if mc_total > 0 else 0
                },
                "fill_in_blank": {
                    "total": fib_total,
                    "correct": fib_correct,
                    "accuracy": fib_correct / fib_total if fib_total > 0 else 0
                },
                "boxed_format": {
                    "total_with_boxed": boxed_total,
                    "correct_with_boxed": boxed_correct,
                    "accuracy_with_boxed": boxed_correct / boxed_total if boxed_total > 0 else 0,
                    "total_without_boxed": non_boxed_total,
                    "correct_without_boxed": non_boxed_correct,
                    "accuracy_without_boxed": non_boxed_correct / non_boxed_total if non_boxed_total > 0 else 0,
                    "boxed_usage_rate": boxed_total / total_count if total_count > 0 else 0
                }
            },
            "detailed_results": results
        }, f, ensure_ascii=False, indent=2)
    
    print("详细结果已保存到 evaluation_results.json")
    
    return final_accuracy, results

if __name__ == "__main__":
    evaluate_benchmark()