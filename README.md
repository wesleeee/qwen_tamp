# Qwen-TAMP: Token-Aware Multimodal Pruning

A token-adaptive pruning framework for Qwen2.5-VL vision-language models with comprehensive evaluation on mathematical reasoning and open-vocabulary detection tasks.

Our work is based on TAMP (https://github.com/G-JWLee/TAMP), VLM-R1 (https://github.com/om-ai-lab/VLM-R1), OVDEval (https://github.com/om-ai-lab/OVDEval) and MATH-Vision (https://github.com/mathllm/MATH-V).

## Overview

This project implements a token-adaptive pruning approach for multimodal large language models, specifically targeting the Qwen2.5-VL model. The framework includes:

- **Token-aware pruning algorithms** for QKV attention weights
- **Mathematical reasoning evaluation** using MATH-V benchmark
- **Open-vocabulary detection evaluation** using OVDEval benchmark
- **CPU and GPU implementations** for different computational requirements

## Architecture

```
qwen_tamp/
├── Core Pruning Modules
│   ├── my_prune.py              # CPU-based pruning implementation
│   ├── my_prune_on_gpu.py       # GPU-accelerated pruning
│   ├── simple_prune.py          # Simple L1 unstructured pruning (baseline)
│   ├── get_weight.py            # QKV weight extraction
│   ├── get_attention.py         # Attention value extraction
│   ├── get_tokens.py            # Token representation extraction
│   └── get_token_source.py      # Vision/text token source identification
├── math-v/MATH-V/               # Mathematical reasoning evaluation
│   ├── data/                    # Test datasets (3,040 problems)
│   ├── images/                  # Mathematical problem images
│   ├── evaluate_benchmark.py    # Evaluation script
│   └── outputs/                 # Evaluation results
├── ovd_eval/OVDEval/           # Open-vocabulary detection evaluation
│   ├── eval_qwen2_5vl.py       # Qwen2.5-VL evaluation script
│   ├── eval_all_datasets_precision_recall.py  # Batch evaluation
│   └── output/                  # Detection results
└── Configuration
    ├── det-exp-virtual.yaml     # Experiment environment config
    └── *.sh                     # Execution scripts
```

## Key Components

### 1. Pruning Algorithms

**Simple Pruning (L1 Unstructured)**
```bash
python simple_prune.py
```
- L1 magnitude-based unstructured pruning
- Configurable sparsity rates

**TAMP Pruning**

```bash
python my_prune_on_gpu.py  # GPU version
python my_prune.py         # CPU version
```
- Attention-guided token importance evaluation
- Cross-modal token correlation analysis
- Adaptive sparsity based on token density

### 2. MATH-V Evaluation

Mathematical reasoning benchmark with 3,040 problems across 16 mathematical subjects.

```bash
cd math-v/MATH-V
python run_evaluation.py
```

### 3. OVDEval Evaluation

Open-vocabulary detection benchmark with 9 subtasks.

```bash
cd ovd_eval/OVDEval
python eval_qwen2_5vl.py --gt-path data/material.json --image-path data/material/ --model-path /path/to/model
```

**Datasets:**
- material, color, position, relationship
- negation, celebrity, logo, landmark

## Installation

```bash
# Create environment
conda create -n qwen_tamp python=3.10
conda activate qwen_tamp

# Install dependencies
pip install torch torchvision transformers vllm qwen-vl-utils
pip install numpy matplotlib scipy tqdm pillow pycocotools
```

## Usage

### Basic Pruning
```bash
# Execute token-aware pruning
python my_prune_on_gpu.py

# Evaluate on MATH-V
cd math-v/MATH-V && python run_evaluation.py

# Evaluate on OVDEval
cd ovd_eval/OVDEval && python eval_qwen2_5vl.py --gt-path data/material.json --image-path data/material/ --model-path /path/to/pruned_model
```

