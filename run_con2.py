# coding=utf-8
import os
import sys
import json
import random
import logging
import argparse
import numpy as np # type: ignore
from datetime import datetime
from tqdm import tqdm
from io import open
from itertools import cycle

import torch # type: ignore
import torch.nn as nn # type: ignore
from torch.utils.data import DataLoader, Dataset, RandomSampler, TensorDataset # type: ignore
from torch.utils.data.distributed import DistributedSampler # type: ignore
import torch.distributed as dist # type: ignore
from torch.nn.parallel import DistributedDataParallel as DDP # type: ignore

from transformers import (
    AutoConfig, AutoModel, AutoTokenizer,
    AdamW, get_linear_schedule_with_warmup
)
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
tau = 12  # 对比学习温度
best_threshold = 0

class Example:
    def __init__(self, code1: str, code2: str, label: int, task: int, lang1: str, lang2: str):
        self.code1 = code1
        self.code2 = code2
        self.label = label
        self.task = task
        self.lang1 = lang1
        self.lang2 = lang2

def read_examples(filename):
    """读取 jsonl 格式样本，返回 Example 列表"""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for  i, line in enumerate(f):
            if i > 500:
                break
            js = json.loads(line)
            code1 = ' '.join(js['Code1'].strip().split())
            code2 = ' '.join(js['Code2'].strip().split())
            label = 1
            task = int(js['Task'])
            lang1 = js['Category1']
            lang2 = js['Category2']
            examples.append(Example(code1, code2, label, task, lang1, lang2))
    return examples

# def convert_examples_to_features(examples, tokenizer, max_length, model_name):
#     features = []
#     for example in examples:
#         encoded1 = tokenizer(
#             example.code1,
#             max_length=max_length,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )
#         encoded2 = tokenizer(
#             example.code2,
#             max_length=max_length,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )
#         # 新增 example.task 到元组中
#         features.append((
#             encoded1['input_ids'], encoded1['attention_mask'],
#             encoded2['input_ids'], encoded2['attention_mask'],
#             example.label,
#             example.task,   # 任务 id
#         ))
#     return features
def convert_examples_to_features(examples, tokenizer, max_length, model_name):
    features = []
    for example in examples:
        encoded1 = tokenizer(
            example.code1,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        encoded2 = tokenizer(
            example.code2,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        features.append((
            encoded1['input_ids'], encoded1['attention_mask'],
            encoded2['input_ids'], encoded2['attention_mask'],
            example.label,  # label
            example.task,   # task id
            example.code1,  # 👈 加入 code1
            example.code2   # 👈 加入 code2
        ))
    return features

def prepare_dataset(examples, tokenizer, max_length, model_name):
    features = convert_examples_to_features(examples, tokenizer, max_length, model_name)

    input_ids1, attention_mask1 = [], []
    input_ids2, attention_mask2 = [], []
    labels = []
    task_ids = []  # 新增任务 id 的列表
    
    for f in features:
        input_ids1.append(f[0].squeeze(0))        # [1,L] -> [L]
        attention_mask1.append(f[1].squeeze(0))
        input_ids2.append(f[2].squeeze(0))
        attention_mask2.append(f[3].squeeze(0))
        labels.append(torch.tensor(f[4], dtype=torch.long))
        task_ids.append(torch.tensor(f[5], dtype=torch.long))  # 加入 task id

    dataset = TensorDataset(
        torch.stack(input_ids1),
        torch.stack(attention_mask1),
        torch.stack(input_ids2),
        torch.stack(attention_mask2),
        torch.stack(labels),
        torch.stack(task_ids)   # 新增到 TensorDataset
    )
    return dataset



# def get_cls_embedding(model, input_ids, attention_mask, model_name):
#     """
#     通用 encoder-only 模型的句向量提取逻辑
#     """
#     model.eval()  # 通常推理时关闭 dropout

#     with torch.no_grad():
#         if 'longformer' in model_name:
#             global_attention_mask = torch.zeros_like(input_ids)
#             global_attention_mask[:, 0] = 1  # 只对 CLS 使用全局 attention
#             outputs = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 global_attention_mask=global_attention_mask
#             )
#         else:
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask)

#     last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
#     cls_embedding = last_hidden_state[:, 0, :]     # [batch_size, hidden_dim]
#     return cls_embedding

def get_sentence_embedding(model, input_ids, attention_mask, model_name, tokenizer=None, debug_batch = None):
    """
    自动适配不同模型结构和封装方式，返回句向量表示（如 [CLS] embedding）

    参数说明：
    - model: 当前模型（可能被 DataParallel 或 DDP 包装）
    - input_ids: 输入 token ids [batch_size, seq_len]
    - attention_mask: attention mask [batch_size, seq_len]
    - model_name: 模型名称（如 "codet5p", "longformer", "unixcoder"...）
    - tokenizer: 对于 longformer 可选（如使用全局 attention）

    返回：
    - Tensor: 句向量 [batch_size, hidden_size]
    """
    #print("input_ids shape:", input_ids.shape)
    #print("attention_mask sum:", attention_mask.sum(dim=1))
    # 解包 DataParallel / DDP 模型
    base_model = model.module if hasattr(model, "module") else model

    if "codet5p" in model_name:
        return base_model(input_ids=input_ids, attention_mask=attention_mask)

    elif "unixcoder" in model_name:
        outputs = base_model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs[0][:, 0, :]  # 取 [CLS]

    elif "longformer" in model_name:
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1  # 只给 CLS 全局 attention

        try:
            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask
            )
            return outputs.last_hidden_state[:, 0, :]  # 取 [CLS]

        except Exception as e:
            error_file = os.path.join("bad_longformer_input.jsonl")
            with open(error_file, "a", encoding="utf-8") as fout:
                batch_size = input_ids.size(0)
                for i in range(batch_size):
                    if debug_batch and len(debug_batch) > i:
                        record = {
                            "code1": debug_batch[i][0],
                            "code2": debug_batch[i][1]
                        }
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            logger.error("❌ Longformer 推理失败，错误样本已保存至 %s", error_file)
            raise e

    else:
        # 通用 HuggingFace encoder-only 模型
        outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]
    
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # ✅ 修正拼写
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
def setup_logger(output_dir):
    log_file_path = os.path.join(output_dir, "log_epoch_train.log")

    # ✅ 避免重复添加 handler（尤其在 DDP 多进程中）
    if any(isinstance(h, logging.FileHandler) and h.baseFilename == log_file_path for h in logger.handlers):
        return

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S'
    ))
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)


def parse_args():
    parser = argparse.ArgumentParser()

    # 必要参数
    #parser.add_argument("--model_type", required=True, type=str)
    parser.add_argument("--model_name_or_path", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--load_model_path", type=str)

    # 数据文件
    parser.add_argument("--train_filename", type=str)
    parser.add_argument("--dev_filename", type=str)
    parser.add_argument("--test_filename", type=str)

    # 模型相关
    parser.add_argument("--config_name", type=str, default="")
    parser.add_argument("--tokenizer_name", type=str, default="")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=32)

    # 训练策略
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_test", action="store_true")

    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--train_steps", type=int, default=-1)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--beam_size", type=int, default=10)

    # 环境
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def init_device(args):
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
    args.device = device
    if args.local_rank in [-1, 0]:
        logger.warning("Device: %s, n_gpu: %d, distributed: %s",
                       device, args.n_gpu, args.local_rank != -1)
    return device

def load_model(args, model_class, config, tokenizer):
    model_name = args.model_name_or_path.lower()
    
    if "codet5p" in model_name:
        model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    elif "longformer" in model_name:
        model = model_class.from_pretrained(args.model_name_or_path, config=config)

    else:
        model = model_class.from_pretrained(args.model_name_or_path, config=config)
    if args.load_model_path is not None:
        if args.local_rank in [-1, 0]:
            logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path, map_location="cpu"))
    model.to(args.device)
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    return model
       
def main():
    dev_dataset = {}

    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.local_rank in [-1, 0]:
        setup_logger(args.output_dir)
        logger.info(f"Arguments: {args}")
    init_device(args)
    config_class, model_class, tokenizer_class = AutoConfig, AutoModel, AutoTokenizer
    model_name = args.model_name_or_path.lower()
    trust_remote = "codet5p" in model_name

    # 加载 tokenizer & config
    config = config_class.from_pretrained(
        args.config_name or args.model_name_or_path,
        trust_remote_code=trust_remote
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name or args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        trust_remote_code=trust_remote
    )
    model = load_model(args, model_class, config, tokenizer)
    if args.do_train:
        
        train_examples = read_examples(args.train_filename)
        #train_dataset = prepare_dataset(train_examples, tokenizer, args.max_source_length, model_name)
        train_dataset= prepare_dataset(train_examples, tokenizer, args.max_source_length, model_name)

        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.train_batch_size // args.gradient_accumulation_steps,
            drop_last=True
        )
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        # Define optimizer_grouped_parameters
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "LayerNorm.weight"])],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in ["bias", "LayerNorm.weight"])],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(t_total * 0.1),
            num_training_steps=t_total
        )
        
        global_step = 0
        if args.local_rank in [-1, 0]:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num epoch = %d", args.num_train_epochs)
        model.train()
        nb_tr_examples, nb_tr_steps,global_step,best_bleu,best_loss = 0,0,0,0,1e6
        for epoch in range(args.num_train_epochs):
            if isinstance(train_sampler, DistributedSampler):
                train_sampler.set_epoch(epoch)
            bar = tqdm(train_dataloader, total=len(train_dataloader)) if args.local_rank in [-1, 0] else train_dataloader
            for step, batch in enumerate(bar):
                
                batch = tuple(t.to(args.device) for t in batch)
                source_ids, source_mask, target_ids, target_mask, labels, task_ids = batch
                unique_tasks = set(t.item() for t in task_ids)
                if len(unique_tasks) == 1:
                    logger.info(f"⚠️ Skipping batch {step}, all task_ids = {list(unique_tasks)}")
                    continue
                #print("Forward start", flush=True)
                
                
                sen_vec1 = get_sentence_embedding(model, source_ids, source_mask, model_name, tokenizer)
                sen_vec2 = get_sentence_embedding(model, target_ids, target_mask, model_name, tokenizer)
            
                
                loss_temp = torch.zeros((len(sen_vec1), len(sen_vec1) * 2 - 1), device=args.device)
           
                for i in range(len(sen_vec1)):
                    loss_temp[i][0] = (nn.CosineSimilarity(dim=0)(sen_vec1[i], sen_vec2[i]) + 1) * 0.5 * tau
                    indice = 1
                    #print("task_ids2:", [t.item() for t in task_ids],flush=True)
                    for j in range(len(sen_vec1)):
                        if i == j:
                            continue
                        temp = j
                        #print("task_ids:", [t.item() for t in task_ids],flush=True)
                        while task_ids[i].item() == task_ids[temp].item():
                            temp = (temp + 1) % len(sen_vec1)
                        loss_temp[i][indice] = (nn.CosineSimilarity(dim=0)(sen_vec1[i], sen_vec2[temp]) + 1) * 0.5 * tau
                        indice += 1
                        loss_temp[i][indice] = (nn.CosineSimilarity(dim=0)(sen_vec1[i], sen_vec1[temp]) + 1) * 0.5 * tau
                        indice += 1
                con_loss = -torch.nn.LogSoftmax(dim=1)(loss_temp)
                loss = torch.sum(con_loss[:, 0]) / len(sen_vec1)
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                bar.set_description(f"epoch {epoch} step {step} loss {loss:.4f}")
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1

    if args.do_eval:
        model.eval()
        #dev_dataset = {}
        current_epoch = epoch if 'epoch' in locals() else 0
        threshold_log_path = os.path.join(args.output_dir, f"thresholds_epoch{current_epoch}.tsv")
       #tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        if 'dev_loss' in dev_dataset:
            eval_examples, eval_data = dev_dataset['dev_loss']
        else:
            eval_examples = read_examples(args.dev_filename)
            eval_data = prepare_dataset(eval_examples, tokenizer, args.max_source_length, model_name)
            dev_dataset['dev_loss'] = (eval_examples, eval_data)

        eval_sampler = RandomSampler(eval_data) if args.local_rank == -1 else DistributedSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=True)
        if args.local_rank in [-1, 0]:
                    logger.info("\n***** Running evaluation *****")
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)
        cos_right, cos_wrong = [], []
        for batch in eval_dataloader:
            batch = tuple(t.to(args.device) for t in batch)
            source_ids, source_mask, target_ids, target_mask, labels, task_ids = batch
            with torch.no_grad():
                sen_vec1 = get_sentence_embedding(model, source_ids, source_mask, model_name, tokenizer)
                sen_vec2 = get_sentence_embedding(model, target_ids, target_mask, model_name, tokenizer)

            cos = nn.CosineSimilarity(dim=1)(sen_vec1, sen_vec2)
            cos_right += cos.tolist()

            for i in range(len(sen_vec1)):
                for j in range(len(sen_vec1)):
                    if i != j and not torch.equal(target_ids[i], target_ids[j]):
                        cos_wrong.append(nn.CosineSimilarity(dim=0)(sen_vec1[i], sen_vec1[j]).item())
                        break

        threshold_log_path = os.path.join(args.output_dir, f"thresholds_epoch{epoch}.tsv")
        best_f1, best_precision, best_recall, best_threshold = 0, 0, 0, 0

        with open(threshold_log_path, "w") as tf:
            tf.write("threshold\trecall\tprecision\tF1\n")
            for i in range(1, 100):
                threshold = i / 100
                tp = sum([1 for s in cos_right if s >= threshold])
                fp = sum([1 for s in cos_wrong if s >= threshold])
                fn = len(cos_right) - tp

                precision = tp / (tp + fp) if tp + fp > 0 else 0
                recall = tp / (tp + fn) if tp + fn > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

                tf.write(f"{threshold:.2f}\t{recall:.4f}\t{precision:.4f}\t{f1:.4f}\n")

                if f1 > best_f1:
                    best_f1 = f1
                    best_precision = precision
                    best_recall = recall
                    best_threshold = threshold

        if args.local_rank in [-1, 0]:
            logger.info(f"[Epoch {epoch}] eval: best F1 = {best_f1:.4f}, precision = {best_precision:.4f}, recall = {best_recall:.4f}, threshold = {best_threshold:.2f}")

        # 保存当前 checkpoint
        checkpoint_dir = os.path.join(args.output_dir, "checkpoint-last")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), os.path.join(checkpoint_dir, f"{epoch}_pytorch_model.bin"))
        if args.local_rank in [-1, 0]:
                    logger.info("  Best F1:%s", best_f1)
                    logger.info("  "+"*"*20)
                    logger.info("  Recall:%s", best_recall)
                    logger.info("  "+"*"*20)
                    logger.info("  Precision:%s", best_precision)
                    logger.info("  "+"*"*20)
                    logger.info("  Best threshold:%s", best_threshold)
                    logger.info("  "+"*"*20)

    if args.do_test:
        model.eval()
        eval_examples = read_examples(args.test_filename)
        eval_data = prepare_dataset(eval_examples, tokenizer, args.max_source_length, model_name)
        eval_sampler = RandomSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=True)

        if args.local_rank in [-1, 0]:
            logger.info("\n***** Running test *****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)

        cos_right, cos_wrong = [], []
        for batch in eval_dataloader:
            batch = tuple(t.to(args.device) for t in batch)
            source_ids, source_mask, target_ids, target_mask, labels, task_ids = batch
            with torch.no_grad():
                sen_vec1 = get_sentence_embedding(model, source_ids, source_mask, model_name, tokenizer)
                sen_vec2 = get_sentence_embedding(model, target_ids, target_mask, model_name, tokenizer)

            cos = nn.CosineSimilarity(dim=1)(sen_vec1, sen_vec2)
            cos_right += cos.tolist()

            for i in range(len(sen_vec1)):
                for j in range(len(sen_vec1)):
                    if i != j and not torch.equal(target_ids[i], target_ids[j]):
                        cos_wrong.append(nn.CosineSimilarity(dim=0)(sen_vec1[i], sen_vec1[j]).item())
                        break

        if not args.do_eval:
            best_threshold = 0.32
            logger.info("using default eval_threshold: %s", best_threshold)
        if args.local_rank in [-1, 0]:
            logger.info("using eval_threshold: %s", best_threshold)

        # 主评估
        tp = sum([1 for s in cos_right if s >= best_threshold])
        fp = sum([1 for s in cos_wrong if s >= best_threshold])
        fn = len(cos_right) - tp

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        logger.info("Test metrics using best threshold")
        result = {'recall': recall, 'precision': precision, 'F1': f1, 'threshold': best_threshold}
        for key in sorted(result.keys()):
            if args.local_rank in [-1, 0]:
                logger.info("  %s = %s", key, str(result[key]))

        # 保存 threshold 曲线
        if args.local_rank in [-1, 0]:
            result_str = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Test: recall={recall:.3f}, precision={precision:.3f}, F1={f1:.3f}, threshold={best_threshold:.2f}\n"
            with open(os.path.join(args.output_dir, 'result'), 'a+') as f:
                f.write(result_str)
            with open('result.txt', 'a+') as f:
                f.write(result_str)
        threshold_log_path = os.path.join(args.output_dir, "thresholds_test.tsv")
        with open(threshold_log_path, "w") as tf:
            tf.write("threshold\trecall\tprecision\tF1\n")
            for i in range(1, 100):
                threshold = i / 100
                tp = sum([1 for s in cos_right if s >= threshold])
                fp = sum([1 for s in cos_wrong if s >= threshold])
                fn = len(cos_right) - tp

                precision = tp / (tp + fp) if tp + fp > 0 else 0
                recall = tp / (tp + fn) if tp + fn > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
                tf.write(f"{threshold:.2f}\t{recall:.4f}\t{precision:.4f}\t{f1:.4f}\n")

        # 写入最终测试结果
        



if __name__ == "__main__":
    main()
                       
                       