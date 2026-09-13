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
save_path = OUTPUT_DIR / "final_checkpoint"
trainer.save_model(str(save_path))
tokenizer.save_pretrained(str(save_path))
print(f"训练完成，适配器已保存到 {save_path}")