#!/usr/bin/env python3
"""
Qwen模型评测运行脚本 - 支持\\boxed{}格式输出
"""

from evaluate_benchmark import evaluate_benchmark

if __name__ == "__main__":
    print("开始运行Qwen模型评测...")
    print("本次评测使用\\boxed{}格式进行答案输出和解析")
    print("请确保以下文件/目录存在：")
    print("- data/test.jsonl")
    print("- images/ 目录及对应的图片文件")
    print()
    
    try:
        accuracy, results = evaluate_benchmark()
        print(f"\n评测成功完成！准确率: {accuracy*100:.2f}%")
        
        # 统计boxed格式使用情况
        boxed_count = sum(1 for r in results if r["has_boxed_format"])
        total_count = len(results)
        if total_count > 0:
            print(f"\\boxed格式使用率: {boxed_count}/{total_count} = {boxed_count/total_count*100:.1f}%")
            
        print("\n提示：")
        print("- 详细结果已保存到 evaluation_results.json")
        print("- 如果\\boxed格式使用率较低，可能需要调整prompt或模型参数")
        
    except Exception as e:
        print(f"评测过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()