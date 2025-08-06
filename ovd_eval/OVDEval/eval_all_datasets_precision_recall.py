#!/usr/bin/env python3
"""
批量评估所有OVDEval数据集的查准率和查全率
计算每个数据集的指标并保存整体平均结果
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any
import subprocess

# 添加当前目录到Python路径，确保可以导入eval_qwen2_5vl模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_single_dataset_evaluation(dataset_name: str, args) -> Dict[str, Any]:
    """
    运行单个数据集的评估
    
    Args:
        dataset_name: 数据集名称
        args: 命令行参数
        
    Returns:
        评估结果字典
    """
    # 构建数据路径
    gt_path = os.path.join(args.data_root, f"{dataset_name}.json")
    image_path = os.path.join(args.data_root, f"{dataset_name}/")
    
    # 检查文件是否存在
    if not os.path.exists(gt_path):
        print(f"⚠️  警告: 找不到数据集文件 {gt_path}")
        return None
        
    if not os.path.exists(image_path):
        print(f"⚠️  警告: 找不到图片目录 {image_path}")
        return None
    
    print(f"\n{'='*60}")
    print(f"🔍 开始评估数据集: {dataset_name}")
    print(f"{'='*60}")
    
    # 构建命令
    cmd = [
        "python", "eval_qwen2_5vl.py",
        "--gt-path", gt_path,
        "--image-path", image_path,
        "--model-path", args.model_path,
        "--output-path", args.output_path,
        "--iou-threshold", str(args.iou_threshold),
        "--conf-threshold", str(args.conf_threshold)
    ]
    
    # 添加可选参数
    if args.max_images:
        cmd.extend(["--max-images", str(args.max_images)])
    if args.max_per_category:
        cmd.extend(["--max-per-category", str(args.max_per_category)])
    if args.sample_ratio:
        cmd.extend(["--sample-ratio", str(args.sample_ratio)])
    
    try:
        # 运行评估脚本
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode != 0:
            print(f"❌ 数据集 {dataset_name} 评估失败:")
            print(f"错误输出: {result.stderr}")
            return None
        
        # 读取评估结果
        eval_result_path = os.path.join(args.output_path, f"{dataset_name}_precision_recall.json")
        
        if os.path.exists(eval_result_path):
            with open(eval_result_path, 'r') as f:
                evaluation_results = json.load(f)
            
            print(f"✅ 数据集 {dataset_name} 评估完成")
            overall = evaluation_results['overall_metrics']
            print(f"   🎯 查准率: {overall['precision']:.4f}")
            print(f"   📈 查全率: {overall['recall']:.4f}")
            print(f"   🏆 F1分数: {overall['f1_score']:.4f}")
            
            return evaluation_results
        else:
            print(f"❌ 找不到评估结果文件: {eval_result_path}")
            return None
            
    except Exception as e:
        print(f"❌ 评估数据集 {dataset_name} 时发生错误: {e}")
        return None

def calculate_average_metrics(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算所有数据集的平均指标
    
    Args:
        all_results: 所有数据集的评估结果列表
        
    Returns:
        平均指标字典
    """
    if not all_results:
        return {}
    
    # 统计总体指标
    total_tp = sum(result['overall_metrics']['tp'] for result in all_results)
    total_fp = sum(result['overall_metrics']['fp'] for result in all_results)
    total_fn = sum(result['overall_metrics']['fn'] for result in all_results)
    
    # 计算宏平均（每个数据集权重相等）
    macro_precision = sum(result['overall_metrics']['precision'] for result in all_results) / len(all_results)
    macro_recall = sum(result['overall_metrics']['recall'] for result in all_results) / len(all_results)
    macro_f1 = sum(result['overall_metrics']['f1_score'] for result in all_results) / len(all_results)
    
    # 计算微平均（每个样本权重相等）
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    
    # 统计详细信息
    total_images = sum(result['detailed_stats']['total_images_processed'] for result in all_results)
    total_gt_images = sum(result['detailed_stats']['images_with_gt'] for result in all_results)
    total_pred_images = sum(result['detailed_stats']['images_with_predictions'] for result in all_results)
    
    return {
        'evaluation_summary': {
            'total_datasets': len(all_results),
            'iou_threshold': all_results[0]['iou_threshold'],
            'evaluation_date': None  # 可以添加时间戳
        },
        'macro_average': {
            'precision': round(macro_precision, 4),
            'recall': round(macro_recall, 4),
            'f1_score': round(macro_f1, 4)
        },
        'micro_average': {
            'precision': round(micro_precision, 4),
            'recall': round(micro_recall, 4),
            'f1_score': round(micro_f1, 4),
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'total_gt': total_tp + total_fn,
            'total_pred': total_tp + total_fp
        },
        'dataset_details': {
            'total_images_processed': total_images,
            'total_images_with_gt': total_gt_images,
            'total_images_with_predictions': total_pred_images
        },
        'individual_results': {
            result.get('dataset_name', f'dataset_{i}'): result['overall_metrics'] 
            for i, result in enumerate(all_results)
        }
    }

def main():
    parser = argparse.ArgumentParser(
        description="批量评估所有OVDEval数据集的查准率和查全率",
        epilog="""
示例用法:
  # 评估所有数据集
  python eval_all_datasets_precision_recall.py --data-root /path/to/OVDEval --model-path /path/to/model --output-path ./output
  
  # 快速测试（每个数据集只处理少量图片）
  python eval_all_datasets_precision_recall.py --data-root /path/to/OVDEval --model-path /path/to/model --max-images 10
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--data-root", type=str, required=True, help="OVDEval数据集根目录")
    parser.add_argument("--model-path", type=str, required=True, help="Qwen2.5-VL模型路径")
    parser.add_argument("--output-path", type=str, default="./output", help="输出目录")
    parser.add_argument("--iou-threshold", type=float, default=0.7, help="IoU阈值")
    parser.add_argument("--conf-threshold", type=float, default=0.9, help="置信度阈值")
    parser.add_argument("--max-images", type=int, default=None, help="每个数据集最大图片数")
    parser.add_argument("--max-per-category", type=int, default=None, help="每个类别最大图片数")
    parser.add_argument("--sample-ratio", type=float, default=None, help="采样比例")
    parser.add_argument("--datasets", nargs='+', default=None, 
                       help="指定要评估的数据集名称，默认评估所有数据集")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    
    # 默认数据集列表
    default_datasets = ["material", "color", "position", "relationship", "negation", "celebrity", "logo", "landmark"]
    
    # 确定要评估的数据集
    datasets_to_evaluate = args.datasets if args.datasets else default_datasets
    
    print(f"🚀 开始批量评估 {len(datasets_to_evaluate)} 个数据集")
    print(f"📁 数据根目录: {args.data_root}")
    print(f"🤖 模型路径: {args.model_path}")
    print(f"💾 输出目录: {args.output_path}")
    print(f"🎯 IoU阈值: {args.iou_threshold}")
    
    if args.max_images:
        print(f"🔢 每个数据集最大图片数: {args.max_images}")
    
    # 评估所有数据集
    all_results = []
    successful_datasets = []
    failed_datasets = []
    
    for dataset_name in datasets_to_evaluate:
        result = run_single_dataset_evaluation(dataset_name, args)
        if result:
            result['dataset_name'] = dataset_name  # 添加数据集名称
            all_results.append(result)
            successful_datasets.append(dataset_name)
        else:
            failed_datasets.append(dataset_name)
    
    # 计算整体平均指标
    if all_results:
        print(f"\n{'='*60}")
        print(f"📊 计算整体平均指标")
        print(f"{'='*60}")
        
        average_results = calculate_average_metrics(all_results)
        
        # 保存整体结果
        overall_save_path = os.path.join(args.output_path, "overall_precision_recall_summary.json")
        with open(overall_save_path, 'w') as f:
            json.dump(average_results, f, indent=2, ensure_ascii=False)
        
        # 打印整体结果
        macro_avg = average_results['macro_average']
        micro_avg = average_results['micro_average']
        
        print(f"📈 宏平均结果 (Macro Average):")
        print(f"   🎯 查准率: {macro_avg['precision']:.4f}")
        print(f"   📈 查全率: {macro_avg['recall']:.4f}")
        print(f"   🏆 F1分数: {macro_avg['f1_score']:.4f}")
        
        print(f"\n📊 微平均结果 (Micro Average):")
        print(f"   🎯 查准率: {micro_avg['precision']:.4f}")
        print(f"   📈 查全率: {micro_avg['recall']:.4f}")
        print(f"   🏆 F1分数: {micro_avg['f1_score']:.4f}")
        print(f"   ✅ 总真正例: {micro_avg['total_tp']}")
        print(f"   ❌ 总假正例: {micro_avg['total_fp']}")
        print(f"   ⚠️  总假负例: {micro_avg['total_fn']}")
        
        print(f"\n💾 整体结果已保存到: {overall_save_path}")
        
        # 成功/失败统计
        print(f"\n📊 评估统计:")
        print(f"   ✅ 成功评估: {len(successful_datasets)} 个数据集")
        if successful_datasets:
            print(f"      {', '.join(successful_datasets)}")
        
        if failed_datasets:
            print(f"   ❌ 失败数据集: {len(failed_datasets)} 个")
            print(f"      {', '.join(failed_datasets)}")
    
    else:
        print("❌ 没有成功评估任何数据集!")
        sys.exit(1)

if __name__ == "__main__":
    main() 