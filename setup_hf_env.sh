#!/bin/bash

conda create -n hf-env python=3.10 -y
conda activate hf-env

pip install transformers datasets requests>=2.32.2

echo "✅ Hugging Face 环境 hf-env 已创建并激活"
echo "使用命令 'conda activate hf-env' 激活环境"
echo "退出环境使用 'conda deactivate'"
