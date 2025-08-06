import json
import argparse
import os
import torch
import numpy as np
import collections
from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import re
from typing import List, Dict, Any



def calculate_iou(box1, boxes2):
    """
    Calculate the intersection ratio (IoU)
    """
    x1, y1, w1, h1 = box1
    x2 = boxes2[:, 0]
    y2 = boxes2[:, 1]
    w2 = boxes2[:, 2]
    h2 = boxes2[:, 3]

    xmin = np.maximum(x1, x2)
    ymin = np.maximum(y1, y2)
    xmax = np.minimum(x1 + w1, x2 + w2)
    ymax = np.minimum(y1 + h1, y2 + h2)

    intersection = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
    union = w1 * h1 + w2 * h2 - intersection

    # Handle cases where the denominator is zero
    iou = np.where(union == 0, 0, intersection / union)

    return iou

def nms_threaded(gt, prediction, iou_threshold, num_threads):
    """NMS处理"""
    if not prediction:
        return []
        
    gt_boxes = np.array([box['bbox'] for box in gt])
    boxes = np.array([box['bbox'] for box in prediction])
    scores = np.array([box['score'] for box in prediction])

    keep_list = []
    remove_list = []
    for idx, i in enumerate(gt):
        gt_box = gt_boxes[idx]
        iou = calculate_iou(gt_box, boxes)
        indices = np.where(iou > iou_threshold)[0].tolist()
        if indices:
            match_scores = scores[indices]
            sorted_indices = np.argsort(match_scores)[::-1]

            final_indices = []
            for sort_idx in sorted_indices.tolist():
                final_indices.append(indices[sort_idx])

            if final_indices:
                keep_list.append(final_indices[0])
                remove_list += final_indices[1:]

    final_remove_list = []
    for i in remove_list:
        if i not in keep_list:
            final_remove_list.append(i)

    selected_boxes = []
    for idx, i in enumerate(prediction):
        if idx not in final_remove_list:
            selected_boxes.append(i)

    return selected_boxes

def load_qwen2_5vl_model(model_path: str):
    """
    加载 Qwen2.5-VL 模型 - 使用 vLLM
    
    Args:
        model_path: 模型权重路径
    
    Returns:
        llm: vLLM 模型实例
        processor: 数据处理器
        sampling_params: 采样参数
    """
    print(f"Loading Qwen2.5-VL model with vLLM from {model_path}...")
    
    # 加载模型 - 按照修改后的 qwen.py 中的方式，添加显存限制
    llm = LLM(
        model=model_path,
        limit_mm_per_prompt={"image": 10, "video": 10},
        gpu_memory_utilization=0.9,
        max_model_len=16384,  # 限制最大序列长度
    )
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.0,  # 设为0确保确定性输出
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=256,
        stop_token_ids=[],
    )
    
    # 加载处理器
    processor = AutoProcessor.from_pretrained(model_path)
    
    print("vLLM model loaded successfully!")
    
    return llm, processor, sampling_params

def parse_bbox_from_text(text: str, image_width: int, image_height: int):
    """
    从模型输出文本中解析边界框坐标
    支持多种格式和坐标系统
    """
    bboxes = []
    
    # 定义多种可能的边界框格式
    patterns = [
        # 标准的 <box> 格式
        r'<box>(\d+),\s*(\d+),\s*(\d+),\s*(\d+)</box>',
        r'<\|box_start\|>(\d+),\s*(\d+),\s*(\d+),\s*(\d+)<\|box_end\|>',
        # 括号格式
        r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]',
        r'\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)',
        # 坐标关键词格式
        r'(?:coordinate|box|bbox|location).*?(\d+),\s*(\d+),\s*(\d+),\s*(\d+)',
        # 简单的四个数字格式（最宽松）
        r'(\d+),\s*(\d+),\s*(\d+),\s*(\d+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            print(f"    🎯 使用模式 '{pattern}' 找到 {len(matches)} 个匹配")
            break
    
    if not matches:
        print(f"    ⚠️  在文本中未找到边界框")
        print(f"    📄 文本内容: {repr(text[:200])}...")
        return bboxes
    
    for i, match in enumerate(matches):
        try:
            x1, y1, x2, y2 = map(int, match)
            # 直接使用解析出的坐标，不进行归一化处理
            print(f"    📍 原始坐标 {i+1}: ({x1}, {y1}, {x2}, {y2})")
            
            # 检查坐标是否需要交换（确保x1<x2, y1<y2）
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            
            # 直接按照 GLIP 的方式处理：假设输入是 [x1, y1, x2, y2] 格式
            w = x2 - x1
            h = y2 - y1
            bbox = [int(x1), int(y1), int(w), int(h)]
            print(f"    📐 转换为 [x,y,w,h] 格式: {bbox}")
            
            # 验证边界框的有效性
            if w > 0 and h > 0 and x1 >= 0 and y1 >= 0:
                bboxes.append(bbox)
                print(f"    ✅ 有效边界框 {i+1}: {bbox}")
            else:
                print(f"    ❌ 无效边界框 {i+1}: {bbox} (w={w}, h={h}, x1={x1}, y1={y1})")
                
        except ValueError as e:
            print(f"    ❌ 坐标解析错误: {match}, 错误: {e}")
            continue
    
    print(f"    📊 最终解析到 {len(bboxes)} 个有效边界框")
    return bboxes

def inference_qwen2_5vl(llm, processor, sampling_params, image_path: str, prompts: List[str], conf_threshold: float = 0.5):
    """
    使用 Qwen2.5-VL 模型进行目标检测推理 - vLLM版本
    
    Args:
        llm: vLLM 模型实例
        processor: 处理器
        sampling_params: 采样参数
        image_path: 图片路径
        prompts: 检测目标列表
        conf_threshold: 置信度阈值
    
    Returns:
        results: 检测结果列表
    """
    image = Image.open(image_path).convert('RGB')
    image_width, image_height = image.size
    results = []
    
    # 为每个目标类别单独进行检测
    for prompt in prompts:
        # 尝试多种不同的提示词格式，找到最有效的
        detection_prompts = [
            f"Detect {prompt} in this image. Provide bounding boxes as <box>x1,y1,x2,y2</box>",
            f"Find all {prompt} objects and give their coordinates in <box>x1,y1,x2,y2</box> format",
            f"Locate {prompt} in the image and return bounding box coordinates",
            f"Can you find {prompt} in this image? Give me the bounding box",
            f"Where is the {prompt}? Provide coordinates"
        ]
        
        found_detection = False
        for detection_prompt in detection_prompts:
            if found_detection:
                break
                
            # 准备输入 - 按照修改后的 qwen.py 中的 vLLM 方式
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                            "min_pixels": 224 * 224,
                            "max_pixels": 1280 * 28 * 28,
                        },
                        {"type": "text", "text": detection_prompt},
                    ],
                }
            ]
            
            # 处理输入 - 按照 vLLM 的方式，使用最兼容的方法
            prompt_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            # 使用兼容的方式处理视觉信息
            try:
                # 首先尝试新版本 API
                image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            except TypeError:
                # 如果新版本失败，使用旧版本 API
                try:
                    image_inputs, video_inputs = process_vision_info(messages)
                    video_kwargs = {}
                except Exception as e:
                    print(f"Warning: process_vision_info failed: {e}")
                    # 如果都失败了，跳过这个图片
                    continue
            
            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs
            
            llm_inputs = {
                "prompt": prompt_text,
                "multi_modal_data": mm_data,
            }
            
            # 只有在有 video_kwargs 时才添加
            if 'video_kwargs' in locals() and video_kwargs:
                llm_inputs["mm_processor_kwargs"] = video_kwargs
            
            # 生成回答 - 使用 vLLM
            outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
            output_text = outputs[0].outputs[0].text
            
            # 打印模型输出用于调试
            print(f"\n🔍 图片: {os.path.basename(image_path)}")
            print(f"📝 检测目标: {prompt}")
            print(f"💬 提示词: {detection_prompt}")
            print(f"🤖 模型输出: {repr(output_text)}")
            print(f"📏 输出长度: {len(output_text)} 字符")
            
            # 解析边界框
            bboxes = parse_bbox_from_text(output_text, image_width, image_height)
            print(f"📦 解析结果: {len(bboxes)} 个边界框")
            
            # 如果找到了边界框，添加到结果中并停止尝试其他提示词
            if bboxes:
                found_detection = True
                for bbox in bboxes:
                    results.append({
                        'bbox': bbox,
                        'score': conf_threshold,  # Qwen2.5-VL 不直接提供置信度，使用阈值作为默认值
                        'label': prompt
                    })
    
    return results

def sample_images_by_category(gt_data, max_per_category=None, sample_ratio=None, max_total=None):
    """
    根据类别采样图片
    
    Args:
        gt_data: 标注数据
        max_per_category: 每个类别最大图片数
        sample_ratio: 采样比例 (0.0-1.0)
        max_total: 总的最大图片数
    
    Returns:
        sampled_images: 采样后的图片列表
    """
    import random
    from collections import defaultdict
    random.seed(7)
    if max_per_category is None and sample_ratio is None and max_total is None:
        return gt_data["images"]
    
    # 按类别分组图片
    category_images = defaultdict(list)
    
    # 创建图片ID到图片信息的映射
    image_id_to_image = {img["id"]: img for img in gt_data["images"]}
    
    # 遍历标注，为每个图片找到对应的类别
    for ann in gt_data["annotations"]:
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        
        if image_id in image_id_to_image:
            img = image_id_to_image[image_id]
            category_images[category_id].append(img)
    
    # 去重（同一张图片可能有多个标注）
    for category_id in category_images:
        seen_ids = set()
        unique_images = []
        for img in category_images[category_id]:
            if img["id"] not in seen_ids:
                unique_images.append(img)
                seen_ids.add(img["id"])
        category_images[category_id] = unique_images
    
    # 打印每个类别的图片数量
    print(f"\n📊 数据采样信息:")
    category_names = {cat["id"]: cat["name"] for cat in gt_data["categories"]}
    
    sampled_images = []
    total_original = len(gt_data["images"])
    
    for category_id, images in category_images.items():
        category_name = category_names.get(category_id, f"Category_{category_id}")
        original_count = len(images)
        
        # 确定采样数量
        if max_per_category:
            sample_count = min(original_count, max_per_category)
        elif sample_ratio:
            sample_count = max(1, int(original_count * sample_ratio))
        else:
            sample_count = original_count
        
        # 随机采样
        if sample_count < original_count:
            sampled = random.sample(images, sample_count)
        else:
            sampled = images
            
        sampled_images.extend(sampled)
        print(f"  📝 {category_name}: {original_count} -> {sample_count} 张图片")
    
    # 去重（防止同一张图片在多个类别中）
    seen_ids = set()
    unique_sampled = []
    for img in sampled_images:
        if img["id"] not in seen_ids:
            unique_sampled.append(img)
            seen_ids.add(img["id"])
    
    # 如果指定了总数限制
    if max_total and len(unique_sampled) > max_total:
        unique_sampled = random.sample(unique_sampled, max_total)
        print(f"  🎯 总数限制: {len(sampled_images)} -> {max_total} 张图片")
    
    print(f"  📈 最终采样: {total_original} -> {len(unique_sampled)} 张图片")
    print(f"  📊 采样比例: {len(unique_sampled)/total_original*100:.1f}%")
    
    return unique_sampled

def calculate_precision_recall(gt_boxes, pred_boxes, iou_threshold=0.7):
    """
    计算查准率(Precision)和查全率(Recall)
    
    Args:
        gt_boxes: 真实标注框列表 [{'bbox': [x,y,w,h], 'category_id': int}, ...]
        pred_boxes: 预测框列表 [{'bbox': [x,y,w,h], 'category_id': int}, ...]
        iou_threshold: IoU阈值
    
    Returns:
        precision: 查准率
        recall: 查全率
        tp: 真正例数量
        fp: 假正例数量
        fn: 假负例数量
    """
    if not pred_boxes and not gt_boxes:
        return 1.0, 1.0, 0, 0, 0  # 都为空时认为完全正确
    
    if not pred_boxes:
        return 0.0, 0.0, 0, 0, len(gt_boxes)  # 没有预测，查准率和查全率都是0
    
    if not gt_boxes:
        return 0.0, 0.0, 0, len(pred_boxes), 0  # 没有真实框，查准率是0
    
    # 转换为numpy数组便于计算
    gt_array = np.array([box['bbox'] for box in gt_boxes])
    pred_array = np.array([box['bbox'] for box in pred_boxes])
    
    # 计算所有预测框与真实框的IoU
    tp = 0  # 真正例
    fp = 0  # 假正例
    matched_gt = set()  # 已匹配的真实框索引
    
    for pred_idx, pred_box in enumerate(pred_boxes):
        pred_bbox = pred_box['bbox']
        best_iou = 0
        best_gt_idx = -1
        
        # 找到与当前预测框IoU最大的真实框
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
                
            gt_bbox = gt_box['bbox']
            
            # 检查类别是否匹配（如果有类别信息）
            if 'category_id' in pred_box and 'category_id' in gt_box:
                if pred_box['category_id'] != gt_box['category_id']:
                    continue
            
            # 计算IoU
            iou = calculate_iou(pred_bbox, np.array([gt_bbox]))[0]
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # 判断是否为真正例
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1
    
    # 计算假负例（未被匹配的真实框）
    fn = len(gt_boxes) - len(matched_gt)
    
    # 计算查准率和查全率
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return precision, recall, tp, fp, fn

def evaluate_dataset_precision_recall(gt_data, pred_results, iou_threshold=0.7):
    """
    评估整个数据集的查准率和查全率
    
    Args:
        gt_data: 真实标注数据
        pred_results: 预测结果列表
        iou_threshold: IoU阈值
    
    Returns:
        dict: 包含总体和各类别的精确率、召回率等指标
    """
    # 按图片ID分组
    gt_by_image = collections.defaultdict(list)
    pred_by_image = collections.defaultdict(list)
    
    # 分组真实标注
    for ann in gt_data['annotations']:
        gt_by_image[ann['image_id']].append({
            'bbox': ann['bbox'],
            'category_id': ann['category_id']
        })
    
    # 分组预测结果
    for pred in pred_results:
        pred_by_image[pred['image_id']].append({
            'bbox': pred['bbox'],
            'category_id': pred['category_id']
        })
    
    # 统计各类别的TP, FP, FN
    category_stats = collections.defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    total_tp, total_fp, total_fn = 0, 0, 0
    
    # 处理每张图片
    all_image_ids = set(gt_by_image.keys()) | set(pred_by_image.keys())
    
    for image_id in all_image_ids:
        gt_boxes = gt_by_image.get(image_id, [])
        pred_boxes = pred_by_image.get(image_id, [])
        
        # 计算该图片的precision和recall
        precision, recall, tp, fp, fn = calculate_precision_recall(
            gt_boxes, pred_boxes, iou_threshold
        )
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # 按类别统计（这里简化处理，实际可以更细化）
        for gt_box in gt_boxes:
            cat_id = gt_box['category_id']
            # 这个框是否被正确检测到，简化判断
            category_stats[cat_id]['fn'] += 1  # 先假设都是FN，后面会修正
            
        for pred_box in pred_boxes:
            cat_id = pred_box['category_id']
            category_stats[cat_id]['fp'] += 1  # 先假设都是FP，后面会修正
    
    # 计算总体指标
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    # 构建类别名称映射
    category_names = {cat['id']: cat['name'] for cat in gt_data['categories']}
    
    results = {
        'iou_threshold': iou_threshold,
        'overall_metrics': {
            'precision': round(overall_precision, 4),
            'recall': round(overall_recall, 4),
            'f1_score': round(overall_f1, 4),
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'total_gt': total_tp + total_fn,
            'total_pred': total_tp + total_fp
        },
        'detailed_stats': {
            'total_images_processed': len(all_image_ids),
            'images_with_gt': len(gt_by_image),
            'images_with_predictions': len(pred_by_image)
        }
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen2.5-VL model on OVDEval",
        epilog="""
示例用法:
  # 每个类别最多处理10张图片
  python eval_qwen2_5vl.py --gt-path data.json --image-path imgs/ --model-path model/ --max-per-category 10
  
  # 随机采样20%的数据
  python eval_qwen2_5vl.py --gt-path data.json --image-path imgs/ --model-path model/ --sample-ratio 0.2
  
  # 总共最多处理50张图片
  python eval_qwen2_5vl.py --gt-path data.json --image-path imgs/ --model-path model/ --max-images 50
  
  # 组合使用：每类最多5张，总共不超过30张
  python eval_qwen2_5vl.py --gt-path data.json --image-path imgs/ --model-path model/ --max-per-category 5 --max-images 30
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gt-path", type=str, required=True, help="Ground truth annotation file path")
    parser.add_argument("--image-path", type=str, required=True, help="Image directory path")
    parser.add_argument("--model-path", type=str, required=True, help="Qwen2.5-VL model path")
    parser.add_argument("--output-path", type=str, default="/home/chuziyuan/zju/qwen_tamp/ovd_eval/OVDEval/output", help="Output directory path")
    parser.add_argument("--iou-threshold", type=float, default=0.7, help="IoU threshold for NMS")
    parser.add_argument("--num-threads", type=int, default=16, help="Number of threads for NMS")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--max-images", type=int, default=None, help="总的最大图片数量限制")
    parser.add_argument("--max-per-category", type=int, default=None, help="每个类别最大图片数 (例如: 10)")
    parser.add_argument("--sample-ratio", type=float, default=None, help="采样比例 0.0-1.0 (例如: 0.2表示20%)")

    args = parser.parse_args()
    
    # 创建输出目录
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    
    # 加载模型
    llm, processor, sampling_params = load_qwen2_5vl_model(args.model_path)
    print("Qwen2.5-VL model with vLLM loaded successfully!")
    
    # 加载标注数据
    gt_data = json.load(open(args.gt_path))
    dataset_name = args.gt_path.rsplit('/', 1)[-1].split('.')[0]
    print(f"==============Now testing {dataset_name}.")
    
    before_nms_results = []
    category_map = {}

    for cat in gt_data["categories"]:
        category_map[cat["name"]] = cat["id"]
    
    # 处理图片 - 智能采样
    print(f"原始数据集包含 {len(gt_data['images'])} 张图片")
    
    # 根据参数进行采样
    images_to_process = sample_images_by_category(
        gt_data, 
        max_per_category=args.max_per_category,
        sample_ratio=args.sample_ratio,
        max_total=args.max_images
    )
    
    if len(images_to_process) != len(gt_data["images"]):
        print(f"✂️  数据采样完成: {len(gt_data['images'])} -> {len(images_to_process)} 张图片")
    
    # 对每张图片进行推理
    total_images = len(images_to_process)
    for idx, img in enumerate(tqdm(images_to_process, desc="Processing images")):
        prompt = []
        img_path = os.path.join(args.image_path, img["file_name"])
        
        # 检查图片是否存在
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        
        # 根据数据集类型确定提示词
        if dataset_name in ['coco', 'celebrity', 'logo', 'landmark']:
            for cat in gt_data["categories"]:
                prompt.append(cat["name"])
        else:
            for t in img["text"] + img["neg_text"]:
                prompt.append(t)

        # 进行推理
        print(f"\n{'='*50}")
        print(f"📸 处理图片 {idx+1}/{total_images}: {img['file_name']}")
        print(f"🎯 检测目标: {prompt}")
        print(f"{'='*50}")
        
        try:
            resp = inference_qwen2_5vl(
                llm, processor, sampling_params, img_path, prompt, args.conf_threshold
            )
            
            # 处理推理结果
            for pred in resp:
                pred_image = {"image_id": img["id"]}
                bbox = pred['bbox']
                
                label = category_map.get(pred['label'], 0)
                score = pred['score']
                
                pred_image["score"] = round(float(score), 8)
                pred_image["category_id"] = label
                pred_image["bbox"] = bbox
                
                before_nms_results.append(pred_image)
            
            print(f"✅ 图片处理完成，检测到 {len(resp)} 个目标")
                
        except Exception as e:
            print(f"❌ 图片处理失败: {img_path}")
            print(f"错误信息: {e}")
            continue

    # NMS 处理
    print("==============Before NMS:", len(before_nms_results))

    gt_data_ann = gt_data['annotations']
    
    image_dict = collections.defaultdict(list)
    gt_dict = collections.defaultdict(list)

    for i in before_nms_results:
        image_dict[i["image_id"]].append(i)

    for i in gt_data_ann:
        gt_dict[i["image_id"]].append(i)
    
    after_nms_results = []
    # 调用多线程并行化 NMS 函数
    for img, preds in image_dict.items():
        gts = gt_dict.get(img, [])
        if gts:  # 只有当存在gt时才进行NMS
            selected_boxes = nms_threaded(gts, preds, args.iou_threshold, args.num_threads)
            after_nms_results += selected_boxes
        else:
            after_nms_results += preds
    
    print("==============After NMS:", len(after_nms_results))
    
    # 计算查准率和查全率
    if after_nms_results:
        print(f"\n==============计算 {dataset_name} 数据集的查准率和查全率")
        
        # 评估当前数据集
        evaluation_results = evaluate_dataset_precision_recall(
            gt_data, after_nms_results, args.iou_threshold
        )
        
        # 保存该数据集的评估结果
        eval_save_path = os.path.join(args.output_path, f"{dataset_name}_precision_recall.json")
        json.dump(evaluation_results, open(eval_save_path, 'w'), indent=2, ensure_ascii=False)
        
        # 打印结果
        overall = evaluation_results['overall_metrics']
        print(f"📊 {dataset_name} 数据集评估结果:")
        print(f"   🎯 查准率 (Precision): {overall['precision']:.4f}")
        print(f"   📈 查全率 (Recall): {overall['recall']:.4f}")
        print(f"   🏆 F1 分数: {overall['f1_score']:.4f}")
        print(f"   ✅ 真正例 (TP): {overall['tp']}")
        print(f"   ❌ 假正例 (FP): {overall['fp']}")
        print(f"   ⚠️  假负例 (FN): {overall['fn']}")
        print(f"   📝 IoU 阈值: {evaluation_results['iou_threshold']}")
        
        # 保存原始检测结果（可选）
        det_save_path = os.path.join(args.output_path, f"{dataset_name}_detections.json")
        json.dump(after_nms_results, open(det_save_path, 'w'), indent=2, ensure_ascii=False)
        
        print(f"\n💾 评估结果已保存到: {eval_save_path}")
        print(f"💾 检测结果已保存到: {det_save_path}")
        
        return evaluation_results  # 返回结果用于后续整体平均计算
    else:
        print("❌ 没有检测结果用于评估!")
        return None

if __name__ == "__main__":
    main() 