# Kaggle AES 2.0 — 基于序数回归与注意力池化的作文自动评分

本项目参加 [Kaggle AES 2.0](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2) 竞赛，使用 **Gemma 4 E4B + LoRA** 微调实现英语作文 1-6 分自动评分。

**最终 OOF QWK：0.8354**（Quadratic Weighted Kappa，二次加权 Kappa 系数）

## 模型架构

```
Gemma 4 E4B（冻结 + LoRA 微调）
  └── Attention Pooling（可学习的全局 token 加权平均）
       └── 序数回归头（Linear 4096→5，5 个独立二分类切点）
            └── BCEWithLogitsLoss + 正类加权（pos_weight）
```

### 四个核心创新

1. **序数回归**：将 1-6 评分转化为 5 个独立二分类（>1? >2? >3? >4? >5?），从根本上解决 Huber/MSE 回归在 OOF 上向均值坍缩的问题
2. **Attention Pooling**：用可学习的跨 token 注意力替代 last-token 池化（零初始化→平滑从均值池化起步→逐步学到有意义的注意力分布）
3. **5-Fold OOF + 动态阈值**：分层 5 折交叉验证产生无偏估计，Nelder-Mead + 暴力搜索优化最优阈值
4. **两阶段渐进训练**：Stage 1 仅训练 head + pooler（暖机）→ Stage 2 解冻 LoRA 联合微调

## 环境配置

### 硬件要求

- Python 3.10+
-  NVIDIA A100 40GB（若 GPU 显存较小，可调小 batch_size）
- CUDA 12.x

### 一键配置

```bash
bash setup.sh
```

### 手动安装

```bash
# 1. 创建虚拟环境
python3.10 -m venv venv
source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载 Gemma 4 E4B 基座模型（如本地没有）
#    huggingface.co/google/gemma-4-4b-it
#    下载后设置环境变量：
export MODEL_PATH=/你的/路径/gemma-4-4b-it

# 4. 数据已包含在 data/ 目录下，直接预处理即可
bash scripts/prepare_data.sh
```

## 数据准备

```bash
# 一键完成：CSV → JSONL + 5-fold 划分
bash scripts/prepare_data.sh

# 等价于手动执行：
# 1. CSV 转 JSONL（train_old.csv → train_persuade.jsonl，train_new.csv → train_kaggle_only.jsonl）
# 2. 分层 5-Fold 划分（按分数分层，确保每折分布一致）
# 3. 输出 train_full.jsonl（含 fold 字段，17,307 条）
```

## 训练流程

### 5-Fold OOF 全流程（一键）

```bash
bash run_v8_kfold.sh
```

自动完成：5 折训练 → 每折 OOF 推理 → 阈值优化

### 单折手动训练

```bash
# 训练 Fold 0
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    stage2_train/sft_ordinal_v8.py \
    --config config/base_config_ordinal_v8.yaml \
    --fold 0 \
    --all_folds data/processed/train_full.jsonl \
    --output_dir outputs/v8_kfold/fold_0 \
    --num_train_epochs 2 --bf16 --flash_attention_2 \
    --optim adamw_torch_fused

# OOF 推理
python stage2_train/predict_oof.py \
    --model_name_or_path outputs/v8_kfold/fold_0/final_regression \
    --base_model_id /你的/路径/gemma-4-E4B-it-local \
    --all_folds data/processed/train_full.jsonl \
    --fold 0 --out_csv outputs/oof/v8_oof.csv

# 阈值优化
python optimize_thresholds_v8.py \
    --oof_csv outputs/oof/v8_oof.csv \
    --output outputs/oof/v8_thresholds.json
```

## 消融实验

训练脚本内置消融开关，可独立控制每个组件：

```bash
# 去掉序数回归（退回 Huber 回归，1-dim 输出）
--no_ordinal

# 去掉注意力池化（退回 Last-Token 池化）
--no_attention_pooling

# 去掉正类加权（普通 BCE）
--no_pos_weight

# 去掉两阶段训练（直接联合训练）
--no_two_stage

# 去掉数据平衡（无样本权重、无过采样）
--no_data_balance
```

完整消融实验脚本见 `ablation/` 目录。

## 实验结果

| 实验 | 池化方式 | 损失函数 | OOF QWK |
|------|---------|---------|------------|
| **V8-Full** | Attention | Ordinal BCE | **0.8354** |
| Exp 2（去掉序数） | Attention | Huber | 0.0239（坍缩） |
| Exp 3（去掉注意力） | Last-Token | Ordinal BCE | 0.249 |
| V6 Baseline | Last-Token | Huber | ~0.23 |

**优化后的阈值**：[1.63, 2.57, 3.59, 4.77, 5.40]

## 项目结构

```
├── stage2_train/
│   ├── sft_ordinal_v8.py      # 核心训练脚本（含所有消融开关）
│   ├── prompt.py               # 统一英文 Prompt 模板
│   └── predict_oof.py          # OOF 推理脚本
├── config/
│   └── base_config_ordinal_v8.yaml  # 训练超参数
├── scripts/
│   ├── prepare_data.sh         # 数据预处理（一键）
│   ├── split_kfold.py          # 分层 5-fold 划分
│   └── split_persuade.py       # Persuade/Kaggle 数据分离
├── ablation/                   # 消融实验完整脚本
│   ├── run_exp2_huber_attn.sh
│   └── run_exp3_ordinal_lasttoken.sh
├── data/                       # 训练数据
│   ├── train_old.csv           # Persuade 语料（12,874 条）
│   └── train_new.csv           # Kaggle 作文（4,433 条）
├── figures/                    # 实验图表 + 生成脚本
│   ├── generate_figures.py
│   └── fig1~6 .png
├── docs/                       # 全套文档
│   ├── 模型训练流程完整文档.md
│   ├── V8完整方案技术详解.md
│   ├── 消融实验与数据图方案.md
│   ├── 智能体使用记录.md
│   ├── 组员贡献分工.md
│   ├── 资源使用情况.md
│   ├── 总结与反思.md
│   └── 博客提纲.md
├── setup.sh                    # 环境配置脚本（一键）
├── optimize_thresholds_v8.py   # 阈值搜索优化
├── vram_utils.py               # GPU 显存工具
├── run_v8_kfold.sh             # 5-fold 全流程
├── run_v8_pipeline.sh          # 两阶段 Persuade→Kaggle 流程
├── requirements.txt            # Python 依赖
└── README.md                   # 本文件
```

## 训练耗时

- **GPU**：4 × NVIDIA A100-SXM4 40GB
- **单折耗时**：约 1.5 小时（Stage 1 ~30min + Stage 2 ~60min）
- **完整 5 折**：约 7.5 小时
- **单卡峰���显存**：约 32 GB（batch_size=2, gradient_checkpointing=True）

## 关键实现细节

### Gemma4 PEFT 兼容性

Gemma4 使用自定义 `ClippableLinear` 层，不暴露标准 `.weight` 属性，导致 PEFT 保存 adapter 时报错。通过 Monkey Patch 动态为该类添加 `.weight` property 解决。

### DDP 多卡训练

- 4 卡 DistributedDataParallel，通过 `torchrun --nproc_per_node=4` 启动
- `device_map={"": f"cuda:{local_rank}"}` 显式绑定 GPU
- NCCL 调优：25MB bucket size，Ring 算法
- `ddp_find_unused_parameters=True` 确保 pooler 模块梯度正确同步

### Prompt 模板

训练和推理使用**完全一致**的英文 `<bos><start_of_turn>user/model` 格式模板，防止 train/inference 分布漂移。

## 参考资料

- Kaggle AES 2.0 赛题：https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2
- Google Gemma 4：https://ai.google.dev/gemma
- Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
- Frank & Hall "A Simple Approach to Ordinal Classification", ECML 2001
- Dao et al. "FlashAttention-2: Faster Attention with Better Parallelism", 2023
