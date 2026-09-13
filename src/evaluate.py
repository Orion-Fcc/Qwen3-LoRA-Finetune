# =====================================================================
# 意图识别评估脚本
# =====================================================================

import json
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix

# ==================== 配置 ====================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_ID = str(BASE_DIR / "models" / "Qwen3-0.6B")
DATA_TRAIN = BASE_DIR / "data" / "train.jsonl"
OUTPUT_DIR = BASE_DIR / "output"
ADAPTER_PATH = OUTPUT_DIR / "final_checkpoint"

CATEGORIES = [
    "weather_query",
    "order_query",
    "cancel_order",
    "refund_request",
    "logistics_query",
    "product_query",
    "account_query",
    "smalltalk",
]

# ==================== 分词器 ====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ==================== 加载基座 + LoRA ====================
print("加载基座模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float16, device_map="auto", trust_remote_code=True,
)
print(f"加载 LoRA 适配器: {ADAPTER_PATH}")
model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
model.eval()

# ==================== 加载数据 ====================
dataset = load_dataset("json", data_files=str(DATA_TRAIN), split="train")
split = dataset.train_test_split(test_size=0.2, seed=42)
test_data = split["test"]
print(f"测试集: {len(test_data)} 条")


# ==================== 推理函数 ====================
def predict(instruction: str, text: str) -> str:
    if text and text.strip():
        prompt = (f"### 指令:\n{instruction}\n\n"
                  f"### 输入:\n{text}\n\n"
                  f"### 回答:\n")
    else:
        prompt = (f"### 指令:\n{instruction}\n\n"
                  f"### 回答:\n")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=10, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### 回答:" in generated:
        pred = generated.split("### 回答:")[-1].strip()
    else:
        pred = generated.strip()
    pred = pred.split("\n")[0].strip()
    return pred if pred in CATEGORIES else "unknown"


# ==================== 逐条评估 ====================
predictions, labels, errors = [], [], []
print("开始评估...")
for i, example in enumerate(test_data):
    pred_label = predict(example["instruction"], example["input"])
    true_label = example["output"]
    predictions.append(pred_label)
    labels.append(true_label)
    if pred_label != true_label:
        errors.append({"text": example["input"], "true": true_label, "pred": pred_label})
    if (i + 1) % 20 == 0:
        print(f"  已评估 {i + 1}/{len(test_data)} 条")

# ==================== 计算指标 ====================
all_labels = CATEGORIES + ["unknown"]
report_dict = classification_report(
    labels, predictions, labels=CATEGORIES,
    output_dict=True, zero_division=0,
)
report_str = classification_report(
    labels, predictions, labels=all_labels,
    zero_division=0, digits=4,
)
cm = confusion_matrix(labels, predictions, labels=all_labels)
accuracy = sum(1 for p, l in zip(predictions, labels) if p == l) / len(labels)

print("\n" + "=" * 60)
print(report_str)
print("=" * 60)
print(f"整体准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"错误样本数: {len(errors)} / {len(labels)}")

# ==================== 保存评估结果 ====================
EVAL_LOG_DIR = OUTPUT_DIR / "eval_results"
EVAL_LOG_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
eval_log_path = EVAL_LOG_DIR / f"eval_{timestamp}.json"

per_class_f1 = {
    label: float(report_dict[label]["f1-score"])
    for label in CATEGORIES if label in report_dict
}

eval_result = {
    "timestamp": timestamp,
    "model_id": MODEL_ID,
    "adapter_path": str(ADAPTER_PATH),
    "test_size": len(test_data),
    "accuracy": accuracy,
    "errors_count": len(errors),
    "classification_report": report_str,
    "confusion_matrix": cm.tolist(),
    "labels": all_labels,
    "per_class_f1": per_class_f1,
    "errors": errors,
}

with open(eval_log_path, "w", encoding="utf-8") as f:
    json.dump(eval_result, f, ensure_ascii=False, indent=2)

print(f"评估结果已保存: {eval_log_path}")