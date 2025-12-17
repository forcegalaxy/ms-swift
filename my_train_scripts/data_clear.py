#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Date    : 2025/07/11 16:23:28
@Author  : Qingxiao Li
@Version : 1.0
@Desc    : 并发跑退火数据过滤


python data_clear.py  --num-processes 64    --load-path "/nfs-13/wuxiaoyu/model_train/ms-swift/my_train_scripts/chem_18018_mat_37088.jsonl"

'''

import os,json,time
from argparse import ArgumentParser
from tqdm import tqdm
import requests, json
import copy, random
import portalocker
import uuid
import hashlib
from multiprocessing import Pool, cpu_count, Manager
from collections import defaultdict
from functools import partial

from typing import List
import re




PROMPT_PREFIX_DICT = {}

def load_jsonl_data(path):
    with open(path,"r",encoding="utf-8") as f:
        lines = [json.loads(each) for each in f.readlines() if each[0]=="{"]
        # print(len(lines), path)
        return lines
    
def load_json_data(path):
    with open(path,"r",encoding="utf-8") as f:
        origin_res = json.load(f)
        # print(len(origin_res), path)
        return origin_res

def dict_to_uuid(d):
    # 将字典转换为排序后的 JSON 字符串
    json_str = json.dumps(d, sort_keys=True)
    # 计算 SHA-256 哈希值
    hash_object = hashlib.sha256(json_str.encode('utf-8'))
    # 基于哈希值生成 UUID
    return str(uuid.UUID(hash_object.hexdigest()[:32]))


def merge_data_and_labele_res(label_dict, data):
    data["response"] = label_dict
    return data


def is_language_consistency(
    question: str,
    reasoning_content: str,
):
    """
    检测推理和回答中的语言一致性，避免中英文混杂
    返回错误列表，并打印匹配到的中文内容
    """
    # errors = []
    # 判断question是否含中文
    has_chinese_in_question = bool(re.search(r'[\u4e00-\u9fff]', question))
    # 仅当语言标签为英文，且question不含中文，才进行混用判断
    if not has_chinese_in_question:
        reasoning_matches = re.findall(r'[\u4e00-\u9fff]+', reasoning_content)
        if reasoning_matches:
            # print(reasoning_matches)
            # print(type(reasoning_matches))
            # print(f"reasoning_content 中匹配到中文: {reasoning_matches}")
            # errors.append(f"reasoning_content 语言混淆：{reasoning_matches}")
            return False, f"出现了语言混淆：{','.join(reasoning_matches)}"
    return True, None



def has_http_url(text):
    """
    检查文本中是否包含 HTTP 或 HTTPS 链接。
    如果包含，则返回 True 和相关信息；否则返回 False 和 None。
    """
    # 使用正则表达式匹配 HTTP 或 HTTPS 链接
    pattern = r'https?://[^\s]+'
    match = re.search(pattern, text)
    
    if match:
        return True, f"包含链接: {match.group(0)}"
    
    return False, None



def overly_reflect(text, overly_reflect_threshold=50):
    """
    检查文本是否包含过多的反思内容。
    如果文本中包含超过 overly_reflect_budget 个 '反思' 相关的词语，则认为是过度反思。
    """
    # overly_reflect_words = ["but", "wait", "But", "Wait", "However", "however", "although", "Although", "yet", "Yet", "still", "Still"]
    overly_reflect_words = ["but", "wait", "But", "Wait"]
    count = sum(text.count(word) for word in overly_reflect_words)
    
    if count > overly_reflect_threshold:
        return True, f"过度反思，包含 {count} 个相关词语:{overly_reflect_words}"
    
    return False, None


def start_with_first(text):
    
    target_invalid_strs = ["First, the question is about", "First, I need to"]
    for string in target_invalid_strs:
        if text.replace("<think>", "").strip().startswith(string):
            return True, f"以First开头的pattern：{string}"
    return False, None



def valid_think_pattern(text, think_budget=100, answer_budget=160):
    # 检查是否只包含一个<think>和一个</think>
    if text.count("<think>") != 1:
        return False, "包含多个 <think> 标签"
    if text.count("</think>") != 1:
        return False, "包含多个 </think> 标签"
        
    elif text.count("<think>") == 1 and text.count("</think>") == 1:
        # 检查<think>是否在</think>之前
        start = text.find("<think>")

        # <think>必须在开头
        if start != 0:
            return False, " <think> 标签不在开头"
        
        end = text.find("</think>")
        # <think>必须在前，且之间有内容
        if start < end and end > start + len("<think>") + think_budget and len(text) - end  > len("</think>") + answer_budget:
            return True, None

    return False, "格式不符合标准<think>...<think>...格式"



def find_repeated_patterns(text, min_len=20, max_len=50, threshold=100):
    '''
    在给定文本中查找重复模式，寻找是否存在一个长度在 min_len到max_len 之间的字串 
    重复模式在全部子串出现超过 threshold 次
    '''
    text_len = len(text)
    
    for size in range(min_len, max_len + 1):
        freq = defaultdict(int)
        
        for i in range(text_len - size + 1):
            sub = text[i:i+size]
            freq[sub] += 1
            
            if freq[sub] == threshold:
                return True, f"重复子串 (长度={size}): {repr(sub)} 已出现超过 {threshold} 次"

    return False, None


def process_data(data, save_path_passed, save_path_unpassed, progress_queue):

    for msg in data['messages']:
        if msg['role'] == 'assistant':
            output_text = msg['content']
        if msg['role'] == 'user':
            input_text = msg['content']

    # need_delete, del_reason = find_repeated_patterns(text=data["output"], min_len=20, max_len=50, threshold=100)
    is_repeat, repeat_reason = find_repeated_patterns(text=output_text, min_len=MIN_LEN, max_len=MAX_LEN, threshold=REPEAT_THRESHOLD)

    # 检查 think 标签
    is_think_valid, think_invalid_reason = valid_think_pattern(text=output_text)

    # 检查是否过度反思
    is_overly_reflect, overly_reflect_reason = overly_reflect(text=output_text, overly_reflect_threshold=OVERLY_REFLECT_THRESHOLD)

    # 检查是否以 First 开头
    is_first_start, first_reason = start_with_first(text=output_text)

    # 检查语言一致性
    is_lang_consistent, lang_consistency_reason = is_language_consistency(
        question=input_text,
        reasoning_content=output_text
    )

    # 检查是否包含链接
    has_link, link_reason = has_http_url(text=output_text)


    # if (not is_repeat) and is_think_valid and (not is_overly_reflect) and (not is_first_start) and (is_lang_consistent):
    if not is_repeat and is_think_valid and not is_overly_reflect and not is_first_start and is_lang_consistent and not has_link:
        
        # 更新 PROMPT_PREFIX_DICT
        prompt_prefix = input_text[:PREFIX_WINDOW_LEN]
        if prompt_prefix in PROMPT_PREFIX_DICT:
            PROMPT_PREFIX_DICT[prompt_prefix] += 1
        else:
            PROMPT_PREFIX_DICT[prompt_prefix] = 1
        
        # 如果以该前缀为开头的数据超过 PREFIX_THRESHOLD 则不要
        if PROMPT_PREFIX_DICT[prompt_prefix] > PREFIX_THRESHOLD:

            with open(save_path_unpassed, 'a', encoding="utf-8") as success_file:
                portalocker.lock(success_file, portalocker.LOCK_EX)
                data["unpass_reason"] = prompt_prefix + f"重复前缀次数超过{PREFIX_THRESHOLD}"
                success_file.write(json.dumps(data, ensure_ascii=False) + '\n')
                success_file.flush()
                portalocker.unlock(success_file)
        
        else:
            with open(save_path_passed, 'a', encoding="utf-8") as success_file:
                portalocker.lock(success_file, portalocker.LOCK_EX)
                success_file.write(json.dumps(data, ensure_ascii=False) + '\n')
                success_file.flush()
                portalocker.unlock(success_file)
    else:
        unpass_reason = ""
        if not is_think_valid:
            unpass_reason += think_invalid_reason + "; "
        if is_repeat:
            unpass_reason += repeat_reason + "; "
        if is_overly_reflect:
            unpass_reason += overly_reflect_reason + "; "
        if is_first_start:
            unpass_reason += first_reason + "; "
        if not is_lang_consistent:
            unpass_reason += lang_consistency_reason + "; "
        if has_link:
            unpass_reason += link_reason + "; "
        with open(save_path_unpassed, 'a', encoding="utf-8") as success_file:
            portalocker.lock(success_file, portalocker.LOCK_EX)
            data["unpass_reason"] = unpass_reason
            success_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            success_file.flush()
            portalocker.unlock(success_file)

    # 更新进度条
    progress_queue.put(1)

def main(load_path, save_path_passed, save_path_unpassed, num_processes):

    # 加载原始数据
    try:
        full_data = load_json_data(load_path)
    except:
        full_data = load_jsonl_data(load_path)

    if not full_data:
        return

    print("数据集原始总量：", len(full_data))

    # 获取需要标注的数据
    # input("Begin! \n>")

    random.shuffle(full_data)

    manager = Manager()
    progress_queue = manager.Queue()


    # 使用多进程进行处理
    with Pool(processes=num_processes) as pool:
        # 使用 `starmap_async`: 返回一个 `MapResult` 对象
        result = pool.starmap_async(process_data, [(data, save_path_passed, save_path_unpassed, progress_queue) for data in full_data])
        
        # 显示进度条
        with tqdm(total=len(full_data)) as pbar:
            # 检查 `result.ready()`: 用于判断所有任务是否完成
            while not result.ready():
                while not progress_queue.empty():
                    # 主进程不断从队列中读取消息并更新进度条
                    progress_queue.get()
                    pbar.update(1)
                # 减少 CPU 占用，同时确保及时更新进度条
                time.sleep(0.1)

if __name__ == "__main__":
    # 
    parser = ArgumentParser(description="check")
    parser.add_argument("-i", "--load-path", type=str, help="待处理文件路径，数据需要 jsonline 的 instruction-input-output")
    parser.add_argument("-n", "--num-processes", type=int, default=64, help="number of processes to use")
    parser.add_argument('-min', '--min-len', type=int, default=20, help="最小重复子串长度")
    parser.add_argument('-max', '--max-len', type=int, default=2000, help="最大重复子串长度")
    # parser.add_argument('-max', '--max-len', type=int, default=50, help="最大重复子串长度")
    parser.add_argument('-rth', '--repeat_threshold', type=int, default=100, help="重复子串出现的阈值")
    parser.add_argument('-pl', '--prefix_window_len', type=int, default=20, help="前缀窗口长度，用于判断重复前缀")
    parser.add_argument('-pth', '--prefix-threshold', type=int, default=1000, help="前缀出现的阈值，超过则不合格")
    parser.add_argument('-oth', '--overly-reflect-threshold', type=int, default=200, help="过度反思的阈值，超过则不合格")
    parser.add_argument('-e', '--echo', action='store_true', default=False, help="是否打印一条合格数据和一条不合格数据的样例")

    args = parser.parse_args()

    # 设置全局变量
    MIN_LEN = args.min_len
    MAX_LEN = args.max_len
    REPEAT_THRESHOLD = args.repeat_threshold
    PREFIX_WINDOW_LEN = args.prefix_window_len
    PREFIX_THRESHOLD = args.prefix_threshold
    OVERLY_REFLECT_THRESHOLD = args.overly_reflect_threshold


    # 验证输入路径是否存在
    if args.load_path is None or not os.path.exists(args.load_path):
        print("*** --load-path not exists. ***")
        exit(0)

    filename = os.path.basename(args.load_path)
    save_path_passed = args.load_path.replace(".json", f"_sanity_rep_min-{MIN_LEN}_max-{MAX_LEN}_th-{REPEAT_THRESHOLD}_pre_len-{PREFIX_WINDOW_LEN}_th-{PREFIX_THRESHOLD}_overef-{OVERLY_REFLECT_THRESHOLD}_keep.json")
    save_path_unpassed = args.load_path.replace(".json", f"_sanity_rep_min-{MIN_LEN}_max-{MAX_LEN}_th-{REPEAT_THRESHOLD}_pre_len-{PREFIX_WINDOW_LEN}_th-{PREFIX_THRESHOLD}_overef-{OVERLY_REFLECT_THRESHOLD}_drop.json")

    # 主程序
    main(
        load_path=args.load_path,
        save_path_passed=save_path_passed, 
        save_path_unpassed=save_path_unpassed,
        num_processes=args.num_processes
    )

    print("Done!")
    

    # 重命名合格数据和不合适数据的文件名，加上数据量
    keep_count = len(load_jsonl_data(save_path_passed))
    drop_count = len(load_jsonl_data(save_path_unpassed))
    print(f"合格数据量: {keep_count}, 不合格数据量: {drop_count}")

    if os.path.exists(save_path_passed):
        os.rename(save_path_passed, save_path_passed.replace(".json", f"_{keep_count}.json"))
    if os.path.exists(save_path_unpassed):
        os.rename(save_path_unpassed, save_path_unpassed.replace(".json", f"_{drop_count}.json"))

    print(f"合格数据保存到: {save_path_passed.replace('.json', f'_{keep_count}.json')}")
    print(f"不合格数据保存到: {save_path_unpassed.replace('.json', f'_{drop_count}.json')}")


    if args.echo:
        print("合格数据样例：")
        keep_case = load_jsonl_data(save_path_passed)
        print(random.choice(keep_case))

        print("不合格数据样例：")
        badcases = load_jsonl_data(save_path_unpassed)
        print(random.choice(badcases))


