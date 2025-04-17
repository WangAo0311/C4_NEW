import pandas as pd
import matplotlib.pyplot as plt

# 文件路径
paths = {
    "codebert": "/home/wangao/Code_clone/c4/C4/results_after_finetune/bert4/thresholds_epoch1.tsv",
    "graphcodebert": "/home/wangao/Code_clone/c4/C4/results_after_finetune/bert4/thresholds_epoch2.tsv",
    "base": "/home/wangao/Code_clone/c4/C4/results_before_fine_tune/baseline/thresholds_test.tsv"
}

plt.figure(figsize=(10, 6))

# 遍历并绘图
for label, path in paths.items():
    df = pd.read_csv(path, sep="\t")

    # 找到最大 F1 对应的 threshold
    max_idx = df["F1"].idxmax()
    max_threshold = df.loc[max_idx, "threshold"]
    max_f1 = df.loc[max_idx, "F1"]

    # 将最大 F1 加到图例标签里
    display_label = f"{label} (F1={max_f1:.3f})"

    # 绘图和标注最大点
    plt.plot(df["threshold"], df["F1"], label=display_label)
    plt.plot(max_threshold, max_f1, "o")  # 圆圈标记最大点

plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.title("F1 vs Threshold Comparison (Epoch 1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("compare_f1_epoch1.png")
print("图像已保存为 compare_f1_epoch1.png")
