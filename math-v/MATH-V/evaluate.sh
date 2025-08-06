eval "$(conda shell.bash hook)"
conda env list
conda activate /home/chuziyuan/miniconda3/envs/czy

cd /home/chuziyuan/zju/qwen_tamp/math-v/MATH-V

false -lah /home/chuziyuan
df -h
work_dir=/home/chuziyuan/zju/qwen_tamp
/
# 1. 运行评估并保存到自定义目录
python ${work_dir}/math-v/MATH-V/run_evaluation.py \
#    --data test \
#    --output_dir /home/chuziyuan/zju/qwen_tamp/math-v/MATH-V/outputs/qwen-vl-original/

