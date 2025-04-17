# test.py
import os
import json
import logging
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from transformers import (
    AutoConfig, AutoModel, AutoTokenizer,
    AdamW, get_linear_schedule_with_warmup
)
from run_con2 import read_examples, prepare_dataset # 根据实际位置修改
from run_con2 import get_sentence_embedding  # 根据实际位置修改
class Example:
    def __init__(self, code1, code2, label, task, idx=None):
        self.code1 = code1
        self.code2 = code2
        self.label = label
        self.task = task
        self.idx = idx
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main():
    model = AutoModel.from_pretrained("allenai/longformer-base-4096")
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    model.to(device)
    config = AutoConfig.from_pretrained("allenai/longformer-base-4096")
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", use_fast=True)
    # 读取 JSONL 文件作为测试样本
    bad_input_path = "/home/wangao/Code_clone/c4/C4/dataset/pair_train.jsonl"
    test_code_pairs = []
    with open(bad_input_path, "r", encoding="utf-8") as f:
         for  i, line in enumerate(f):
            if i > 1000:
                break
            item = json.loads(line)
            code1 = item.get("code1", "").replace("\\n", "\n")
            code2 = item.get("code2", "").replace("\\n", "\n")
            test_code_pairs.append((code1, code2))

    # 构造 Example 类的 list
    test_examples = []
    for idx, (c1, c2) in enumerate(test_code_pairs):
        test_examples.append(Example(code1=c1, code2=c2, label=1, task=0, idx=idx))

    # 编码成 dataset（注意 prepare_dataset 返回两个）
    test_dataset, test_code_pairs = prepare_dataset(test_examples, tokenizer,1024, "allenai/longformer-base-4096")

    test_sampler = RandomSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=8,
        drop_last=False
    )

    logger.info("***** Running testing on bad_longformer_input.jsonl *****")
    logger.info("  Num examples = %d", len(test_examples))
    

    model.eval()
    with torch.no_grad():
        bar = tqdm(test_dataloader, total=len(test_dataloader))
        for step, batch in enumerate(bar):
            batch = tuple(t.to(device) for t in batch)
            source_ids, source_mask, target_ids, target_mask, labels, task_ids = batch

            # unique_tasks = set(t.item() for t in task_ids)
            # if len(unique_tasks) == 1:
            #     print(f"⚠️ Skipping batch {step}, all task_ids = {list(unique_tasks)}", flush=True)
            #     continue

            if hasattr(test_sampler, 'indices'):
                batch_indices = test_sampler.indices[step * 8 : (step + 1) * 8]
            else:
                batch_indices = list(range(step * 8, (step + 1) *8))

            debug_batch = [test_code_pairs[i] for i in batch_indices]

            try:
                sen_vec1 = get_sentence_embedding(model, source_ids, source_mask, "allenai/longformer-base-4096", tokenizer, debug_batch)
                sen_vec2 = get_sentence_embedding(model, target_ids, target_mask, "allenai/longformer-base-4096", tokenizer, debug_batch)

                # 这里只是测试，可以打印余弦相似度
                cos_sim = torch.nn.functional.cosine_similarity(sen_vec1, sen_vec2)
                print("Cosine Similarity:", cos_sim.cpu().numpy())

            except Exception as e:
                logger.error("Exception in batch %d: %s", step, str(e))
                continue

if __name__ == "__main__":
    main()
