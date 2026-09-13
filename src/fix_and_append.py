# =====================================================================
# 数据修正 + 边界样本追加
# =====================================================================

import json
import shutil
from pathlib import Path
from collections import Counter

# ==================== 配置 ====================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw.jsonl"
BACKUP_PATH = BASE_DIR / "data" / "raw_backup.jsonl"

# 需要修正的样本：{文本: 新标签}
FIX_LABELS = {
    "我买的那面镜子，物流显示已经到小区了，但没收到任何取件通知，我该去哪拿？": "logistics_query",
    "我临时要搬家，地址全变了，货到了我也收不着了。": "cancel_order",
}

# 追加的边界样本：(文本, 标签)
NEW_SAMPLES = [
    # ---------- order_query：强调订单本身 ----------
    ("我买的东西发货了没", "order_query"),
    ("刚下的单咋没动静昂", "order_query"),
    ("订单状态显示异常怎么办", "order_query"),
    ("我订单里的商品能改吗", "order_query"),
    ("订单详情界面打不开，你帮我看看", "order_query"),
    ("这个号是不是我下的单,看下", "order_query"),
    ("这笔订单的金额是多少", "order_query"),
    ("我付款了怎么订单还显示待付款", "order_query"),
    ("订单什么时候能发货", "order_query"),
    ("帮我查下我的订单记录", "order_query"),
    ("我买的两件东西是一单吗", "order_query"),
    ("订单能查到具体下单时间吗", "order_query"),
    ("我订单里有几个商品", "order_query"),
    ("订单状态一直不变是什么原因", "order_query"),
    ("刚付完款订单生成了吗", "order_query"),

    # ---------- logistics_query：强调包裹运输 ----------
    ("啥时候能派到我这儿呀，等得花儿都谢了", "logistics_query"),
    ("都等五天了，东西还在路上晃悠呢", "logistics_query"),
    ("那家店说发货了，可我这边的记录还停在揽收那一步，是不是光打了个单子没给货啊？", "logistics_query"),
    ("发货了没，多久能到哇", "logistics_query"),
    ("我的包裹是不是已经到配送点了", "logistics_query"),
    ("物流信息停在昨天了，咋不更了", "logistics_query"),
    ("快递到没到驿站，我下班去取", "logistics_query"),
    ("送货的到哪儿了，能查不", "logistics_query"),
    ("这快递也忒慢了，到底啥时候能到啊", "logistics_query"),
    ("我的包裹现在走到哪儿了呀", "logistics_query"),
    ("我买的鞋显示已揽收，怎么就不动了呢", "logistics_query"),
    ("客服在吗，帮我查下快递走到哪个环节了", "logistics_query"),
    ("送快递的今天会来吗", "logistics_query"),
    ("能不能给催催，太急用这个了", "logistics_query"),
    ("快递小哥大概几点送啊，家里得留人", "logistics_query"),

    # ---------- weather_query：防止被判成 smalltalk ----------
    ("最近是不是一直阴天啊，好烦", "weather_query"),
    ("这会外面的风大不大，我刚洗完头", "weather_query"),
    ("重庆这温度咋跟火炉似的，啥时候凉快点", "weather_query"),
    ("外面闷不闷", "weather_query"),
    ("今天空气湿度大不大", "weather_query"),
    ("晚上会不会降温，我怕冷", "weather_query"),
    ("明天早上有雾吗，开车怕看不清", "weather_query"),
    ("今天紫外线强不强，要不要涂防晒", "weather_query"),
    ("这周气温是不是一直挺高的", "weather_query"),
    ("外面出太阳了吗", "weather_query"),

    # ---------- smalltalk：防止被天气类抢走 ----------
    ("周末打算去爬山，可天气预报说可能要下雨。", "smalltalk"),
    ("最近怎么老下雨，烦死了。", "smalltalk"),
    ("这破网速，卡死我了", "smalltalk"),
    ("今天心情特别好", "smalltalk"),
    ("刚吃了碗面，撑死我了", "smalltalk"),
    ("你咋这么逗呢", "smalltalk"),
    ("哎，你说人活着图啥呢", "smalltalk"),
    ("刚睡醒，脑子还懵着呢", "smalltalk"),
    ("谢啦兄弟", "smalltalk"),
    ("哈哈，笑死我了", "smalltalk"),
    ("这破天儿，热得受不了", "smalltalk"),
    ("跟你说话挺开心的", "smalltalk"),
]


def main():
    # 备份
    if not BACKUP_PATH.exists():
        shutil.copy(DATA_RAW, BACKUP_PATH)
        print(f"已备份: {BACKUP_PATH}")

    # 读取
    records = []
    with open(DATA_RAW, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"读取原始数据: {len(records)} 条")

    # 修正
    fixed = 0
    for item in records:
        text = item.get("text", "")
        if text in FIX_LABELS:
            old = item["label"]
            new = FIX_LABELS[text]
            if old != new:
                item["label"] = new
                print(f"[修正] {old} -> {new} | {text[:40]}")
                fixed += 1
    print(f"共修正 {fixed} 条")

    # 追加
    existing = {r.get("text", "") for r in records}
    added = 0
    for text, label in NEW_SAMPLES:
        if text in existing:
            continue
        records.append({"text": text, "label": label})
        added += 1
    print(f"共追加 {added} 条")

    # 写回
    with open(DATA_RAW, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"处理完成，共 {len(records)} 条")

    # 统计
    counter = Counter(r["label"] for r in records)
    print("\n类别分布:")
    for label, count in sorted(counter.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()