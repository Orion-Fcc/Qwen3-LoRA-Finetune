# =====================================================================
# 数据预处理：raw.jsonl -> train.jsonl
# =====================================================================

import json
from pathlib import Path
from collections import Counter

# ==================== 配置 ====================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw.jsonl"
DATA_TRAIN = BASE_DIR / "data" / "train.jsonl"

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

SYSTEM_PROMPT = f"""你是一个意图分类器。

可选类别：
{", ".join(CATEGORIES)}

分类规则：
- 天气、气温、下雨、天气预报：weather_query
- 查询订单状态：order_query
- 取消订单：cancel_order
- 退款、退钱：refund_request
- 快递、物流、配送进度：logistics_query
- 商品信息、价格、库存：product_query
- 登录、账号、密码：account_query
- 问候、闲聊：smalltalk

注意区分：
- order_query：问订单本身（下单、付款、发货、订单记录）
- logistics_query：问包裹运输（快递到哪了、派送、签收）
- cancel_order：还没发货，想撤销订单
- refund_request：已付款/已收货，要求退钱

只输出一个类别名称，不要输出解释。""".strip()


# ==================== 主流程 ====================
def main():
    records = []
    with open(DATA_RAW, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"第 {i} 行 JSON 格式错误: {e}")
                continue

            text = item.get("text", "").strip()
            label = item.get("label", "").strip()
            if not text or not label:
                continue
            if label not in CATEGORIES:
                print(f"第 {i} 行未知类别: {label}")
                continue

            records.append({
                "instruction": SYSTEM_PROMPT,
                "input": text,
                "output": label,
            })

    with open(DATA_TRAIN, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"转换完成：{len(records)} 条 -> {DATA_TRAIN}")

    counter = Counter(r["output"] for r in records)
    print("\n类别分布:")
    for label, count in sorted(counter.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()