import json
from collections import defaultdict
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import os

# Model name
model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
max_length = 512
print(f"Using model: {model_name}, max token length: {max_length}")

# File paths
file_paths = [
    "/home/wangao/Code_clone/c4/C4/dataset/pair_test.jsonl",
    "/home/wangao/Code_clone/c4/C4/dataset/pair_valid.jsonl",
    "/home/wangao/Code_clone/c4/C4/dataset/pair_train.jsonl"
]

# Initialize counters
total_code_snippets = 0
code_over_512 = 0

# Token length bin size
bin_size = 128
length_distribution = defaultdict(int)

# Process each file
for file_path in file_paths:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            for code in [item.get("Code1", ""), item.get("Code2", "")]:
                token_len = len(tokenizer(code, truncation=False)["input_ids"])
                total_code_snippets += 1

                if token_len > 512:
                    code_over_512 += 1

                bin_index = (token_len // bin_size) * bin_size
                length_distribution[f"{bin_index}-{bin_index + bin_size - 1}"] += 1

# Print statistics
print(f"\nTotal code snippets (including Code1 and Code2): {total_code_snippets}")
print(f"Code snippets with length > 512: {code_over_512} ({code_over_512 / total_code_snippets:.2%})")

# Print distribution table
print("\nToken length distribution:")
sorted_bins = sorted(length_distribution.keys(), key=lambda x: int(x.split('-')[0]))
for bin_range in sorted_bins:
    count = length_distribution[bin_range]
    print(f"{bin_range}: {count} ({count / total_code_snippets:.2%})")

# ----------- Plotting section -----------
# Prepare data
import matplotlib.ticker as ticker

# Prepare data
bin_labels = sorted(length_distribution.keys(), key=lambda x: int(x.split('-')[0]))
counts = [length_distribution[br] for br in bin_labels]
bin_midpoints = [int(br.split('-')[0]) + bin_size // 2 for br in bin_labels]

# Calculate proportions
gt_512 = sum(count for br, count in length_distribution.items() if int(br.split('-')[0]) >= 512)
gt_1024 = sum(count for br, count in length_distribution.items() if int(br.split('-')[0]) >= 1024)

# Plot
plt.figure(figsize=(14, 6))
bars = plt.bar(bin_midpoints, counts, width=bin_size * 0.9, align='center')

# Add lines at 512 and 1024
plt.axvline(x=512, color='red', linestyle='--', label='Token Length = 512')
plt.axvline(x=1024, color='green', linestyle='--', label='Token Length = 1024')

# Add text for proportions
plt.text(520, max(counts) * 0.85, f">{512}: {gt_512} ({gt_512 / total_code_snippets:.2%})", color='red')
plt.text(1030, max(counts) * 0.75, f">{1024}: {gt_1024} ({gt_1024 / total_code_snippets:.2%})", color='green')

# Aesthetics
plt.xticks(rotation=45)
plt.xlabel("Token Length Range (centered)")
plt.ylabel("Number of Code Snippets")
plt.title("Token Length Distribution of Code Snippets")
plt.xlim(0, 2048)  # Limit x axis for better centering
plt.ylim(0, max(counts) * 1.2)  # Add headroom

plt.legend()
plt.grid(axis='y', linestyle=':', linewidth=0.5)
plt.tight_layout()

# Save
output_path = "token_length_distribution_marked.png"
plt.savefig(output_path)
print(f"\nUpdated plot saved to: {os.path.abspath(output_path)}")

