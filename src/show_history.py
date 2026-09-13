# =====================================================================
# 展示所有训练和评估历史 + 可视化曲线
# =====================================================================

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==================== 配置 ====================
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
TRAIN_LOG_DIR = OUTPUT_DIR / "train_logs"
EVAL_LOG_DIR = OUTPUT_DIR / "eval_results"

# =====================================================================
# 第一部分：文本历史记录
# =====================================================================

print("=" * 90)
print("训练历史")
print("=" * 90)

train_files = sorted(TRAIN_LOG_DIR.glob("train_*.json")) if TRAIN_LOG_DIR.exists() else []
train_logs = []

if train_files:
    print(f"共 {len(train_files)} 次训练\n")
    for i, f in enumerate(train_files, 1):
        with open(f, "r", encoding="utf-8") as fp:
            log = json.load(fp)
        train_logs.append(log)

        lc = log["lora_config"]
        ta = log["training_args"]
        ds = log["dataset"]

        print(f"[{i}] {log['timestamp']}")
        print(f"    LoRA: r={lc['r']}, alpha={lc['lora_alpha']}, dropout={lc['lora_dropout']}")
        print(f"    训练: lr={ta['learning_rate']}, epoch={ta['num_train_epochs']}, "
              f"batch={ta['per_device_train_batch_size']}×{ta['gradient_accumulation_steps']}")
        print(f"    数据: 训练 {ds['train_size']} | 验证 {ds['eval_size']} | 测试 {ds['test_size']}")

        history = log.get("log_history", [])
        tl = [e["loss"] for e in history if "loss" in e]
        el = [e["eval_loss"] for e in history if "eval_loss" in e]
        if tl:
            print(f"    训练 loss: {tl[0]:.4f} -> {tl[-1]:.4f}")
        if el:
            print(f"    验证 loss: {el[0]:.4f} -> {el[-1]:.4f}")
        print()
else:
    print("暂无训练记录")

# ==================== 评估历史 ====================
print("\n" + "=" * 90)
print("评估历史")
print("=" * 90)

eval_files = sorted(EVAL_LOG_DIR.glob("eval_*.json")) if EVAL_LOG_DIR.exists() else []
eval_logs = []

if eval_files:
    print(f"共 {len(eval_files)} 次评估\n")
    print(f"{'时间':<22} {'准确率':<12} {'错误数':<10} {'最低F1类':<20} {'最低F1':<10}")
    print("-" * 90)

    for f in eval_files:
        with open(f, "r", encoding="utf-8") as fp:
            log = json.load(fp)
        eval_logs.append(log)

        acc = log.get("accuracy", 0)
        err = log.get("errors_count", 0)
        f1 = log.get("per_class_f1", {})

        if f1:
            min_label = min(f1, key=f1.get)
            min_f1 = f1[min_label]
        else:
            min_label, min_f1 = "N/A", 0

        print(f"{log['timestamp']:<22} {acc:<12.4f} {err:<10} {min_label:<20} {min_f1:<10.4f}")

    if len(eval_files) >= 2:
        print("\n" + "=" * 90)
        print("最近两次对比")
        print("=" * 90)
        prev, curr = eval_logs[-2], eval_logs[-1]
        print(f"上次: {prev['timestamp']}  准确率 {prev['accuracy']:.4f}")
        print(f"这次: {curr['timestamp']}  准确率 {curr['accuracy']:.4f}")
        delta = curr["accuracy"] - prev["accuracy"]
        print(f"变化: {'+' if delta >= 0 else ''}{delta:.4f} ({delta*100:+.2f}%)")

        print("\n每类 F1 对比:")
        print(f"{'类别':<20} {'上次':<10} {'这次':<10} {'变化':<10}")
        print("-" * 60)
        pf, cf = prev.get("per_class_f1", {}), curr.get("per_class_f1", {})
        for label in pf:
            p, c = pf[label], cf.get(label, 0)
            print(f"{label:<20} {p:<10.4f} {c:<10.4f} {c-p:+.4f}")
else:
    print("暂无评估记录")


# =====================================================================
# 第二部分：可视化曲线
# =====================================================================

if not train_logs and not eval_logs:
    print("\n没有数据可以画图")
    exit()

# ==================== 图 1：最新一次训练的详细曲线 ====================
if train_logs:
    latest = train_logs[-1]
    history = latest.get("log_history", [])

    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []
    lr_steps, lr_values = [], []
    grad_steps, grad_values = [], []
    epoch_steps, epoch_values = [], []

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
        if "epoch" in entry:
            epoch_steps.append(step)
            epoch_values.append(entry["epoch"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"训练详细曲线 - {latest['timestamp']}", fontsize=14)

    # Loss 曲线
    if train_losses:
        axes[0, 0].plot(train_steps, train_losses, label="训练 Loss", color="blue")
    if eval_losses:
        axes[0, 0].plot(eval_steps, eval_losses, label="验证 Loss",
                        color="red", marker="o", markersize=4)
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

    # 学习率
    if lr_values:
        axes[0, 1].plot(lr_steps, lr_values, color="orange")
    axes[0, 1].set_xlabel("训练步数")
    axes[0, 1].set_ylabel("Learning Rate")
    axes[0, 1].set_title("学习率变化")
    axes[0, 1].grid(alpha=0.3)

    # 梯度范数
    if grad_values:
        axes[1, 0].plot(grad_steps, grad_values, color="purple")
    axes[1, 0].set_xlabel("训练步数")
    axes[1, 0].set_ylabel("Grad Norm")
    axes[1, 0].set_title("梯度范数")
    axes[1, 0].grid(alpha=0.3)

    # Epoch vs Loss
    if epoch_values and train_losses:
        axes[1, 1].scatter(epoch_values[:len(train_losses)], train_losses,
                           color="teal", alpha=0.6, s=20)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].set_title("Epoch vs Loss")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = OUTPUT_DIR / f"train_curve_{latest['timestamp']}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n训练曲线已保存: {save_path}")
    plt.show()


# ==================== 图 2：多次评估对比 ====================
if len(eval_logs) >= 2:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("多次评估对比", fontsize=14)

    timestamps = [log["timestamp"] for log in eval_logs]
    accuracies = [log["accuracy"] for log in eval_logs]

    # 整体准确率
    axes[0].plot(range(len(eval_logs)), accuracies,
                 marker="o", color="blue", linewidth=2, markersize=8)
    for i, acc in enumerate(accuracies):
        axes[0].annotate(f"{acc:.4f}", (i, acc),
                         textcoords="offset points", xytext=(0, 10),
                         ha="center", fontsize=9)
    axes[0].axhline(y=0.90, color="green", linestyle="--", alpha=0.5, label="目标 90%")
    axes[0].set_xticks(range(len(eval_logs)))
    axes[0].set_xticklabels(timestamps, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("准确率")
    axes[0].set_title("整体准确率变化")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 每类 F1
    all_labels = list(eval_logs[0].get("per_class_f1", {}).keys())
    for label in all_labels:
        f1_values = [log.get("per_class_f1", {}).get(label, 0) for log in eval_logs]
        axes[1].plot(range(len(eval_logs)), f1_values,
                     marker="o", label=label, linewidth=1.5)
    axes[1].set_xticks(range(len(eval_logs)))
    axes[1].set_xticklabels(timestamps, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title("每类 F1 变化")
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "eval_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"评估对比图已保存: {save_path}")
    plt.show()


# ==================== 图 3：多次训练 Loss 对比 ====================
if len(train_logs) >= 2:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("多次训练 Loss 对比", fontsize=14)

    colors = plt.cm.tab10.colors

    for i, log in enumerate(train_logs):
        history = log.get("log_history", [])
        steps = [e["step"] for e in history if "loss" in e]
        losses = [e["loss"] for e in history if "loss" in e]
        eval_steps_ = [e["step"] for e in history if "eval_loss" in e]
        eval_losses_ = [e["eval_loss"] for e in history if "eval_loss" in e]

        color = colors[i % len(colors)]
        axes[0].plot(steps, losses, label=f"{log['timestamp']} train",
                     color=color, linewidth=1.5)
        if eval_losses_:
            axes[1].plot(eval_steps_, eval_losses_,
                         label=f"{log['timestamp']} eval",
                         color=color, linewidth=1.5, marker="o", markersize=3)

    axes[0].set_xlabel("训练步数")
    axes[0].set_ylabel("训练 Loss")
    axes[0].set_title("训练 Loss 对比")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("训练步数")
    axes[1].set_ylabel("验证 Loss")
    axes[1].set_title("验证 Loss 对比")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "train_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"训练对比图已保存: {save_path}")
    plt.show()