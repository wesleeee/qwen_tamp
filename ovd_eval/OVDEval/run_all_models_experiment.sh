#!/bin/bash

eval "$(conda shell.bash hook)"
conda env list
conda activate /home/chuziyuan/miniconda3/envs/czy

ls -lah /home/chuziyuan
df -h
work_dir=/home/chuziyuan/zju/qwen_tamp/ovd_eval/OVDEval

# Qwen2.5-VL OVDEval 多模型完整实验脚本
# 作者: Assistant
# 用法: bash run_all_models_experiment.sh

echo "======================================================"
echo "        Qwen2.5-VL 多模型 OVDEval 完整实验"
echo "======================================================"

# 定义模型和输出路径数组
declare -a MODEL_PATHS=(
    "/home/chuziyuan/zju/TAMP/checkpoints/VLM-R1-Qwen2.5VL-3B-OVD-0321/"
    "/home/chuziyuan/zju/qwen_tamp/simple_pruned_qwen_model/"
    "/home/chuziyuan/zju/qwen_tamp/my_pruned_qwen_model/"
)

declare -a OUTPUT_ROOTS=(
    "/home/chuziyuan/zju/qwen_tamp/ovd_eval/OVDEval/no_output"
    "/home/chuziyuan/zju/qwen_tamp/ovd_eval/OVDEval/simple_output"
    "/home/chuziyuan/zju/qwen_tamp/ovd_eval/OVDEval/my_output"
)

declare -a MODEL_NAMES=(
    "VLM-R1-Qwen2.5VL-3B-OVD"
    "simple_pruned_qwen"
    "my_pruned_qwen"
)

# 公共设置
DATA_ROOT="/home/chuziyuan/zju/qwen_tamp/data/OVDEval"
datasets=("material" "color" "position" "relationship" "negation" "celebrity" "logo" "landmark")

# 创建总的汇总文件
TOTAL_SUMMARY_FILE="/home/chuziyuan/zju/qwen_tamp/ovd_eval/OVDEval/all_models_summary.txt"
echo "Qwen2.5-VL 多模型 OVDEval 评测结果汇总" > "$TOTAL_SUMMARY_FILE"
echo "评测时间: $(date)" >> "$TOTAL_SUMMARY_FILE"
echo "========================================" >> "$TOTAL_SUMMARY_FILE"
echo "" >> "$TOTAL_SUMMARY_FILE"

# 检查数据集路径
if [ ! -d "$DATA_ROOT" ]; then
    echo "❌ 错误: 数据集路径不存在: $DATA_ROOT"
    exit 1
fi

# 记录总开始时间
TOTAL_START_TIME=$(date +%s)

# 循环处理每个模型
for i in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH="${MODEL_PATHS[$i]}"
    OUTPUT_ROOT="${OUTPUT_ROOTS[$i]}"
    MODEL_NAME="${MODEL_NAMES[$i]}"
    
    echo ""
    echo "======================================================"
    echo "开始测试模型 $((i+1))/3: $MODEL_NAME"
    echo "======================================================"
    echo "模型路径: $MODEL_PATH"
    echo "输出路径: $OUTPUT_ROOT"
    echo ""
    
    # 检查模型路径是否存在
    if [ ! -d "$MODEL_PATH" ]; then
        echo "❌ 错误: 模型路径不存在: $MODEL_PATH"
        echo "模型 $MODEL_NAME: 路径不存在，跳过" >> "$TOTAL_SUMMARY_FILE"
        continue
    fi
    
    # 创建输出目录
    echo "创建输出目录..."
    mkdir -p "$OUTPUT_ROOT"
    echo "✅ 输出目录创建完成: $OUTPUT_ROOT"
    
    # 创建模型专用汇总文件
    MODEL_SUMMARY_FILE="$OUTPUT_ROOT/model_evaluation_summary.txt"
    echo "模型 $MODEL_NAME 评测结果汇总" > "$MODEL_SUMMARY_FILE"
    echo "评测时间: $(date)" >> "$MODEL_SUMMARY_FILE"
    echo "模型路径: $MODEL_PATH" >> "$MODEL_SUMMARY_FILE"
    echo "========================================" >> "$MODEL_SUMMARY_FILE"
    echo "" >> "$MODEL_SUMMARY_FILE"
    
    # 记录模型开始时间
    MODEL_START_TIME=$(date +%s)
    
    # 记录到总汇总文件
    echo "========================================" >> "$TOTAL_SUMMARY_FILE"
    echo "模型: $MODEL_NAME" >> "$TOTAL_SUMMARY_FILE"
    echo "路径: $MODEL_PATH" >> "$TOTAL_SUMMARY_FILE"
    echo "开始时间: $(date)" >> "$TOTAL_SUMMARY_FILE"
    echo "========================================" >> "$TOTAL_SUMMARY_FILE"
    
    # 逐个评测每个数据集
    successful_datasets=0
    failed_datasets=0
    
    for dataset in "${datasets[@]}"; do
        echo "------------------------------------------------------"
        echo "正在评测数据集: $dataset (模型: $MODEL_NAME)"
        echo "------------------------------------------------------"
        
        # 检查数据集文件是否存在
        ANNOTATION_FILE="$DATA_ROOT/${dataset}.json"
        IMAGE_DIR="$DATA_ROOT/${dataset}/"
        
        if [ ! -f "$ANNOTATION_FILE" ]; then
            echo "❌ 警告: 标注文件不存在: $ANNOTATION_FILE"
            echo "数据集 $dataset: 标注文件不存在" >> "$MODEL_SUMMARY_FILE"
            echo "数据集 $dataset: 标注文件不存在" >> "$TOTAL_SUMMARY_FILE"
            ((failed_datasets++))
            continue
        fi
        
        if [ ! -d "$IMAGE_DIR" ]; then
            echo "❌ 警告: 图片目录不存在: $IMAGE_DIR"
            echo "数据集 $dataset: 图片目录不存在" >> "$MODEL_SUMMARY_FILE"
            echo "数据集 $dataset: 图片目录不存在" >> "$TOTAL_SUMMARY_FILE"
            ((failed_datasets++))
            continue
        fi
        
        # 创建数据集专用输出目录
        DATASET_OUTPUT="$OUTPUT_ROOT/${dataset}"
        mkdir -p "$DATASET_OUTPUT"
        
        # 运行评测
        DATASET_START_TIME=$(date +%s)
        
        echo "开始时间: $(date)"
        python ${work_dir}/eval_qwen2_5vl.py \
            --gt-path "$ANNOTATION_FILE" \
            --image-path "$IMAGE_DIR" \
            --model-path "$MODEL_PATH" \
            --output-path "$DATASET_OUTPUT" \
            --iou-threshold 0.2 \
            --conf-threshold 0.9 \
            --max-per-category 1 \
            2>&1 | tee "$DATASET_OUTPUT/${dataset}_eval_log.txt"
        
        EVAL_EXIT_CODE=$?
        DATASET_END_TIME=$(date +%s)
        DATASET_DURATION=$((DATASET_END_TIME - DATASET_START_TIME))
        
        echo "结束时间: $(date)"
        echo "用时: ${DATASET_DURATION} 秒"
        
        # 记录到汇总文件
        if [ $EVAL_EXIT_CODE -eq 0 ]; then
            echo "✅ 数据集 $dataset 评测完成"
            echo "数据集 $dataset: 评测成功, 用时 ${DATASET_DURATION} 秒" >> "$MODEL_SUMMARY_FILE"
            echo "数据集 $dataset: 评测成功, 用时 ${DATASET_DURATION} 秒" >> "$TOTAL_SUMMARY_FILE"
            ((successful_datasets++))
            
            # 提取主要结果 - 查找精确率和召回率
            if [ -f "$DATASET_OUTPUT/${dataset}_eval_log.txt" ]; then
                echo "  主要结果:" >> "$MODEL_SUMMARY_FILE"
                echo "  主要结果:" >> "$TOTAL_SUMMARY_FILE"
                
                # 提取查准率和查全率信息
                grep -E "(查准率|查全率|F1|Precision|Recall)" "$DATASET_OUTPUT/${dataset}_eval_log.txt" >> "$MODEL_SUMMARY_FILE" 2>/dev/null || true
                grep -E "(查准率|查全率|F1|Precision|Recall)" "$DATASET_OUTPUT/${dataset}_eval_log.txt" >> "$TOTAL_SUMMARY_FILE" 2>/dev/null || true
            fi
        else
            echo "❌ 数据集 $dataset 评测失败"
            echo "数据集 $dataset: 评测失败" >> "$MODEL_SUMMARY_FILE"
            echo "数据集 $dataset: 评测失败" >> "$TOTAL_SUMMARY_FILE"
            ((failed_datasets++))
        fi
        
        echo "" >> "$MODEL_SUMMARY_FILE"
        echo "" >> "$TOTAL_SUMMARY_FILE"
        echo ""
    done
    
    # 记录模型总时间
    MODEL_END_TIME=$(date +%s)
    MODEL_DURATION=$((MODEL_END_TIME - MODEL_START_TIME))
    
    echo "======================================================"
    echo "模型 $MODEL_NAME 评测完成!"
    echo "======================================================"
    echo "成功评测: $successful_datasets 个数据集"
    echo "失败评测: $failed_datasets 个数据集"
    echo "用时: ${MODEL_DURATION} 秒 ($(($MODEL_DURATION / 60)) 分钟)"
    echo "结果保存在: $OUTPUT_ROOT"
    echo ""
    
    # 记录到总汇总
    echo "模型 $MODEL_NAME 完成: 成功 $successful_datasets, 失败 $failed_datasets, 用时 ${MODEL_DURATION} 秒" >> "$TOTAL_SUMMARY_FILE"
    echo "" >> "$TOTAL_SUMMARY_FILE"

done

# 记录总时间
TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))

echo "======================================================"
echo "全部实验完成!"
echo "======================================================"
echo "测试了 ${#MODEL_PATHS[@]} 个模型"
echo "总用时: ${TOTAL_DURATION} 秒 ($(($TOTAL_DURATION / 60)) 分钟)"
echo "总汇总报告: $TOTAL_SUMMARY_FILE"
echo ""

# 显示总汇总
echo "======================================================"
echo "全部评测汇总:"
echo "======================================================"
cat "$TOTAL_SUMMARY_FILE"

echo ""
echo "🎉 全部实验已完成! 查看详细结果请检查各个输出目录中的文件。"

# 可选：运行批量评估脚本生成整体平均结果
echo ""
echo "🔄 正在为每个模型生成整体平均结果..."

for i in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH="${MODEL_PATHS[$i]}"
    OUTPUT_ROOT="${OUTPUT_ROOTS[$i]}"
    MODEL_NAME="${MODEL_NAMES[$i]}"
    
    echo "生成 $MODEL_NAME 的整体平均结果..."
    
    python ${work_dir}/eval_all_datasets_precision_recall.py \
        --data-root "$DATA_ROOT" \
        --model-path "$MODEL_PATH" \
        --output-path "$OUTPUT_ROOT" \
        --max-per-category 1 \
        --datasets material color position relationship negation celebrity logo landmark \
        2>&1 | tee "$OUTPUT_ROOT/overall_evaluation_log.txt"
    
    if [ $? -eq 0 ]; then
        echo "✅ $MODEL_NAME 整体评估完成"
    else
        echo "❌ $MODEL_NAME 整体评估失败"
    fi
done

echo ""
echo "🎊 所有评估(包括整体平均)已完成!" 