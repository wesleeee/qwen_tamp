#!/usr/bin/env python3
"""
简单的 Qwen2.5-VL 模型测试脚本
用于验证模型加载和推理是否正常工作
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from eval_qwen2_5vl import load_qwen2_5vl_model, inference_qwen2_5vl, parse_bbox_from_text
import argparse

def test_model_loading(model_path):
    """测试模型加载"""
    print("Testing model loading...")
    try:
        llm, processor, sampling_params = load_qwen2_5vl_model(model_path)
        print("✓ Model loaded successfully!")
        return llm, processor, sampling_params
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return None, None, None

def test_inference(llm, processor, sampling_params, image_path, prompts):
    """测试推理功能"""
    print("Testing inference...")
    try:
        results = inference_qwen2_5vl(llm, processor, sampling_params, image_path, prompts)
        print(f"✓ Inference successful! Found {len(results)} detections")
        for i, result in enumerate(results):
            print(f"  Detection {i+1}: {result}")
        return results
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return []

def test_bbox_parsing():
    """测试边界框解析"""
    print("Testing bbox parsing...")
    test_text = "I found a cat at <box>100,200,300,400</box> and a dog at <box>500,600,700,800</box>"
    bboxes = parse_bbox_from_text(test_text, 1000, 1000)
    print(f"✓ Parsed {len(bboxes)} bounding boxes: {bboxes}")
    return bboxes

def main():
    parser = argparse.ArgumentParser(description="Test Qwen2.5-VL evaluation script")
    parser.add_argument("--model-path", type=str, help="Path to Qwen2.5-VL model",default="/home/chuziyuan/zju/TAMP/checkpoints/VLM-R1-Qwen2.5VL-3B-OVD-0321")
    parser.add_argument("--image-path", type=str, help="Path to test image (optional)", 
                       default="/home/chuziyuan/zju/qwen_tamp/test_image/蓝桥杯.jpg")
    
    args = parser.parse_args()
    
    print("=== Qwen2.5-VL Model Test ===")
    print(f"Model path: {args.model_path}")
    print(f"Image path: {args.image_path or 'Not provided'}")
    print()
    
    # 测试边界框解析
    test_bbox_parsing()
    print()
    
    # 测试模型加载
    llm, processor, sampling_params = test_model_loading(args.model_path)
    if llm is None:
        print("Stopping tests due to model loading failure.")
        return
    
    print()
    
    # 如果提供了图片，测试推理
    if args.image_path and os.path.exists(args.image_path):
        test_prompts = ["cup", "computer", "person", "book"]
        test_inference(llm, processor, sampling_params, args.image_path, test_prompts)
    else:
        print("No valid image path provided, skipping inference test.")
    
    print("\n=== Test completed ===")

if __name__ == "__main__":
    main() 