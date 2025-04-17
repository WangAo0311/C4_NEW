# test_longformer.py

import torch
import json
from transformers import AutoTokenizer, AutoModel
from argparse import ArgumentParser
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_longformer_cls_embedding(model, input_ids, attention_mask):
    assert input_ids.device == attention_mask.device
    assert attention_mask.dtype == torch.long
    assert input_ids.dtype == torch.long

    logger.info(f"📌 input_ids.shape = {input_ids.shape}")
    logger.info(f"📌 attention_mask.shape = {attention_mask.shape}")
    logger.info(f"📌 attention_mask.sum(dim=1) = {attention_mask.sum(dim=1)}")

    # sliding_attention_mask 就用 attention_mask 原样
    sliding_attention_mask = attention_mask.clone()

    # global attention：只对每行的第一个位置设为 1
    global_attention_mask = torch.zeros_like(attention_mask)
    global_attention_mask[:, 0] = 1

    # 不推荐全局 attention 掩码全为 1，会影响 longformer 的 sliding attention 效果

    logger.info(f"📌 sliding_attention_mask.sum(dim=1) = {sliding_attention_mask.sum(dim=1)}")
    logger.info(f"📌 global_attention_mask.sum(dim=1) = {global_attention_mask.sum(dim=1)}")

    model_to_use = model.module if hasattr(model, "module") else model

    try:
        outputs = model_to_use(
            input_ids=input_ids,
            attention_mask=sliding_attention_mask,
            global_attention_mask=global_attention_mask,
            output_hidden_states=False
        )
    except Exception as e:
        import traceback
        logger.error("🔥 Exception traceback:\n" + traceback.format_exc())
        logger.error(f"🔥 Longformer forward failed. input_ids.shape = {input_ids.shape}")
        logger.error(f"🔥 input_ids[0][-10:] = {input_ids[0][-10:].tolist()}")
        logger.error(f"🔥 attention_mask.sum(dim=1) = {attention_mask.sum(dim=1).tolist()}")
        logger.error(f"🔥 sliding_attention_mask.sum(dim=1) = {sliding_attention_mask.sum(dim=1).tolist()}")
        logger.error(f"🔥 global_attention_mask.sum(dim=1) = {global_attention_mask.sum(dim=1).tolist()}")
        raise e

    return outputs[0][:, 0, :]

class Example(object):
    def __init__(self, source1, source2, target, lang1, lang2):
        self.source = [source1, source2]
        self.lang = [lang1, lang2]
        self.target = target

language_dict = {}
def get_catagory_id(catagory):
    if catagory not in language_dict:
        language_dict[catagory] = len(language_dict)
    return language_dict[catagory]

def read_examples(filename):
    examples = []
    with open(filename, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > 50:
                break
            line = line.strip()
            js = json.loads(line)
            code1 = ' '.join(js['Code1'].replace('\n', ' ').strip().split())
            code2 = ' '.join(js['Code2'].replace('\n', ' ').strip().split())
            task = ' '.join(str(js['Task']).replace('-', ' ').split())
            lang1 = get_catagory_id(js['Category1'])
            lang2 = get_catagory_id(js['Category2'])
            examples.append(Example(source1=code1, source2=code2, target=task, lang1=lang1, lang2=lang2))
    return examples
def source_process(tokenizer, args, example, i):
    # 让 tokenizer 自动处理截断、加CLS/SEP、padding
    encoded = tokenizer(
            example.source[i],
            max_length=2048,
            padding='max_length',
            truncation=True,
            return_tensors=None  # 返回list而不是tensor
        )
    # logger.info(f"[source_process] example.source[{i}][:100] = {example.source[i][:100]}")
    # logger.info(f"[source_process] Encoded input_ids len = {len(encoded['input_ids'])}, should ≤ {args.max_source_length}")
    # logger.info(f"[source_process] input_ids[:10] = {encoded['input_ids'][:10]}")
   #logger.info(f"[source_process] Encoded input_ids len = {len(encoded['input_ids'])}")
    
    return encoded['input_ids'], encoded['attention_mask']

def target_process(tokenizer, args, example):
    target_tokens = example.target[:args.max_target_length-2]
    target_ids = [int(target_tokens)]
    target_mask = [1] *len(target_ids)
    padding_length = args.max_target_length - len(target_ids)
    
    return target_ids,target_mask
class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 source_ids1,
                 source_ids2,
                 target_ids1,
                 target_ids2,
                 source_mask1,
                 source_mask2,
                 target_mask1,
                 target_mask2,
                 lang1,
                 lang2,
    ):
        self.source_ids1 = source_ids1
        self.source_ids2 = source_ids2
        self.target_ids1 = target_ids1
        self.target_ids2 = target_ids2
        self.source_mask1 = source_mask1
        self.source_mask2 = source_mask2
        self.target_mask1 = target_mask1
        self.target_mask2 = target_mask2
        self.source_lang1 = lang1
        self.source_lang2 = lang2   
def convert_examples_to_features(examples, tokenizer, args, portion=1):
    features = []
    for example_index, example in enumerate(examples):
        #source
        source_ids1, source_mask1 = source_process(tokenizer, args, example, 0)
        source_ids2, source_mask2 = source_process(tokenizer, args, example, 1)
 
        #target
        target_ids1, target_mask1 = target_process(tokenizer, args, example)   
        target_ids2 = target_ids1
        target_mask2 = target_mask1
       
        features.append(
            InputFeatures(
                 source_ids1,
                 source_ids2,
                 target_ids1,
                 target_ids2,
                 source_mask1,
                 source_mask2,
                 target_mask1,
                 target_mask2,
                 example.lang[0],
                 example.lang[1],
            )
        )
    features = features[:int(len(features)*portion)]
    return features
def prepare_dataset(args, filename, tokenizer, portion=1):
    train_examples = read_examples(filename)
    train_features = convert_examples_to_features(train_examples, tokenizer,args, portion)
    all_source_ids1 = torch.tensor([f.source_ids1 for f in train_features], dtype=torch.long)
    all_source_mask1 = torch.tensor([f.source_mask1 for f in train_features], dtype=torch.long)
    all_target_ids1 = torch.tensor([f.target_ids1 for f in train_features], dtype=torch.long)
    all_target_mask1 = torch.tensor([f.target_mask1 for f in train_features], dtype=torch.long)   
    all_source_ids2 = torch.tensor([f.source_ids2 for f in train_features], dtype=torch.long)
    all_source_mask2 = torch.tensor([f.source_mask2 for f in train_features], dtype=torch.long)
    all_target_ids2 = torch.tensor([f.target_ids2 for f in train_features], dtype=torch.long)
    all_target_mask2 = torch.tensor([f.target_mask2 for f in train_features], dtype=torch.long) 
    train_data = TensorDataset(all_source_ids1, all_source_mask1, all_target_ids1, all_target_mask1, all_source_ids2, all_source_mask2, all_target_ids2, all_target_mask2)
    return train_examples,train_data   


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", default="allenai/longformer-base-4096")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path)
    model.eval().cuda()

    examples = read_examples(args.input_file)
    logger.info(f"Loaded {len(examples)} examples")
    train_examples, train_data = prepare_dataset(args, args.train_filename, tokenizer, 1)
    train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size//args.gradient_accumulation_steps, drop_last=True)
    bar = tqdm(train_dataloader, total=len(train_dataloader))
    for idx, batch in enumerate(bar):
        # batch = tuple(t.to(device) for t in batch)
        if args.n_gpu > 1:
            batch = tuple(t.to(model.device_ids[0]) for t in batch)
        else:
            batch = tuple(t.to(device) for t in batch)
        source_ids,source_mask,task1_ids,_,target_ids,target_mask,_,_ = batch
    for i, example in enumerate(examples):
        tokens1 = tokenizer(
            example.source[0], max_length=args.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        tokens2 = tokenizer(
            example.source[1], max_length=args.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        input_ids1 = tokens1["input_ids"].cuda()
        attention_mask1 = tokens1["attention_mask"].cuda()
        input_ids2 = tokens2["input_ids"].cuda()
        attention_mask2 = tokens2["attention_mask"].cuda()

        with torch.no_grad():
            sen_vec1 = get_longformer_cls_embedding(model, input_ids1, attention_mask1)
            sen_vec2 = get_longformer_cls_embedding(model, input_ids2, attention_mask2)

        cos_sim = torch.nn.functional.cosine_similarity(sen_vec1, sen_vec2).item()
        print(f"[Sample {i}] Cosine similarity = {cos_sim:.4f}")

if __name__ == "__main__":
    main()
