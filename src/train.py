# =====================================================================
# Qwen3-0.6B LoRA 微调训练脚本（数据加载优化版）
# =====================================================================

# ==================== 导入依赖 ====================
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ==================== 路径配置 ====================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "train.jsonl"
OUTPUT_DIR = BASE_DIR / "output"

# ==================== 模型配置 ====================
MODEL_ID = str(BASE_DIR / "models" / "Qwen3-0.6B")

# ==================== 加载分词器 ====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ==================== 4-bit 量化配置 ====================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# ==================== 加载模型 ====================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# ==================== 准备 k-bit 训练 ====================
model = prepare_model_for_kbit_training(model)

# ==================== 配置 LoRA ====================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==================== 加载并切分数据 ====================
dataset = load_dataset("json", data_files=str(DATA_PATH), split="train")

split = dataset.train_test_split(test_size=0.2, seed=42)
train_val = split["train"].train_test_split(test_size=0.125, seed=42)

train_data = train_val["train"]
eval_data = train_val["test"]
test_data = split["test"]

print(f"训练集: {len(train_data)} 条 | 验证集: {len(eval_data)} 条 | 测试集: {len(test_data)} 条")

# ==================== 格式化数据 ====================
def format_prompt(example):
    if example.get("input") and example["input"].strip():
        text = (
            f"### 指令:\n{example['instruction']}\n\n"
            f"### 输入:\n{example['input']}\n\n"
            f"### 回答:\n{example['output']}"
        )
    else:
        text = (
            f"### 指令:\n{example['instruction']}\n\n"
            f"### 回答:\n{example['output']}"
        )
    return {"text": text}

train_data = train_data.map(format_prompt)
eval_data = eval_data.map(format_prompt)
test_data = test_data.map(format_prompt)

# ==================== 训练配置 ====================
training_args = SFTConfig(
    output_dir=str(OUTPUT_DIR),

    # ---------- 批次大小（不变） ----------
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,

    # ---------- 训练轮次（不变） ----------
    num_train_epochs=3,

    # ---------- 学习率（不变） ----------
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=0.1,

    # ---------- 日志与评估（不变） ----------
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=20,
    save_steps=50,
    save_total_limit=2,
    dataset_text_field="text",

    # ---------- 显存优化 ----------
    optim="adamw_8bit",                                          # 【改动 1】
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},      # 【改动 2】

    # ---------- 序列长度（不变） ----------
    max_length=256,

    # ---------- 数据加载加速 ----------
    dataloader_num_workers=4,                                    # 【改动 3】
    dataloader_pin_memory=True,                                  # 【改动 4】
    dataloader_persistent_workers=True,                          # 【改动 5】

    # ---------- 其他（不变） ----------
    packing=False,
    report_to="none",
)

# ==================== 开始训练 ====================
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    processing_class=tokenizer,
)

trainer.train()

# ==================== 保存 LoRA 适配器 ====================
import json
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

save_path = OUTPUT_DIR / "final_checkpoint"
trainer.save_model(str(save_path))
tokenizer.save_pretrained(str(save_path))
print(f"训练完成，适配器已保存到 {save_path}")

# ==================== 保存训练参数（带时间戳） ====================
TRAIN_LOG_DIR = OUTPUT_DIR / "train_logs"
TRAIN_LOG_DIR.mkdir(parents=True, exist_ok=True)
train_log_path = TRAIN_LOG_DIR / f"train_{timestamp}.json"

train_log = {
    "timestamp": timestamp,
    "model_id": MODEL_ID,
    "data_path": str(DATA_PATH),
    "lora_config": {
        "r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
        "target_modules": list(lora_config.target_modules),
        "bias": lora_config.bias,
        "task_type": str(lora_config.task_type),
    },
    "training_args": {
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "num_train_epochs": training_args.num_train_epochs,
        "learning_rate": training_args.learning_rate,
        "lr_scheduler_type": training_args.lr_scheduler_type,
        "warmup_steps": training_args.warmup_steps,
        "optim": training_args.optim,
        "bf16": training_args.bf16,
        "gradient_checkpointing": training_args.gradient_checkpointing,
        "max_length": training_args.max_length,
        "packing": training_args.packing,
    },
    "dataset": {
        "train_size": len(train_data),
        "eval_size": len(eval_data),
        "test_size": len(test_data),
    },
    "log_history": trainer.state.log_history,
}

with open(train_log_path, "w", encoding="utf-8") as f:
    json.dump(train_log, f, ensure_ascii=False, indent=2)

print(f"训练参数已保存: {train_log_path}")

# ==================== 保存本次训练的曲线图 ====================
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

history = trainer.state.log_history

train_steps, train_losses = [], []
eval_steps, eval_losses = [], []
lr_steps, lr_values = [], []
grad_steps, grad_values = [], []

for entry in history:
    step = entry.get("step")
    if step is None:
        continue
    if "loss" in entry:
        train_steps.append(step)
        train_losses.append(entry["loss"])
    if "eval_loss" in entry:
        eval_steps.append(step)
        eval_losses.append(entry["eval_loss"])
    if "learning_rate" in entry:
        lr_steps.append(step)
        lr_values.append(entry["learning_rate"])
    if "grad_norm" in entry:
        grad_steps.append(step)
        grad_values.append(entry["grad_norm"])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"训练曲线 - {timestamp}", fontsize=14)

# ---- 子图 1：Loss 曲线 ----
if train_losses:
    axes[0, 0].plot(train_steps, train_losses, label="训练 Loss", color="blue", linewidth=1.5)
if eval_losses:
    axes[0, 0].plot(eval_steps, eval_losses, label="验证 Loss",
                    color="red", marker="o", markersize=4, linewidth=1.5)
    min_eval = min(eval_losses)
    min_step = eval_steps[eval_losses.index(min_eval)]
    axes[0, 0].axvline(x=min_step, color="green", linestyle="--", alpha=0.5)
    axes[0, 0].annotate(f"最低 {min_eval:.4f}\nstep {min_step}",
                        xy=(min_step, min_eval),
                        xytext=(min_step + 20, min_eval + 0.2),
                        fontsize=9,
                        arrowprops=dict(arrowstyle="->", color="green"))
axes[0, 0].set_xlabel("训练步数")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title("Loss 曲线")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# ---- 子图 2：学习率 ----
if lr_values:
    axes[0, 1].plot(lr_steps, lr_values, color="orange", linewidth=1.5)
axes[0, 1].set_xlabel("训练步数")
axes[0, 1].set_ylabel("Learning Rate")
axes[0, 1].set_title("学习率变化")
axes[0, 1].grid(alpha=0.3)

# ---- 子图 3：梯度范数 ----
if grad_values:
    axes[1, 0].plot(grad_steps, grad_values, color="purple", linewidth=1.2)
axes[1, 0].set_xlabel("训练步数")
axes[1, 0].set_ylabel("Grad Norm")
axes[1, 0].set_title("梯度范数")
axes[1, 0].grid(alpha=0.3)

# ---- 子图 4：Epoch vs Loss ----
epochs = [e.get("epoch") for e in history if "loss" in e and "epoch" in e]
losses = [e["loss"] for e in history if "loss" in e and "epoch" in e]
if epochs and losses:
    axes[1, 1].scatter(epochs, losses, color="teal", alpha=0.6, s=20)
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Loss")
axes[1, 1].set_title("Epoch vs Loss")
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
curve_path = OUTPUT_DIR / f"train_curve_{timestamp}.png"
plt.savefig(curve_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"训练曲线已保存: {curve_path}")