"""
WMT中英数据预处理脚本
将CSV格式转换为训练所需的文本格式

CSV格式说明:
- 第0行可能是header: "0,1"
- 从第1行开始是数据，格式: "中文内容,英文内容"
- 英文部分可能用引号包围
"""

import re
import os
import argparse


def convert_wmt_data(csv_file, output_zh, output_en, sample_limit=None):
    """
    直接读取原始行，手动解析
    """
    print(f"Loading {csv_file}...")
    
    zh_texts = []
    en_texts = []
    error_count = 0
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            # 跳过可能的header行
            if idx == 0 and line == "0,1":
                print("跳过header行")
                continue
            
            # 处理Windows行结束符
            line = line.replace('\r\n', '').replace('\r', '')
            
            # 解析: 找到最后一个逗号之前是中文，之后是英文
            # 但如果英文被引号包围，需要特殊处理
            
            # 方法1: 尝试匹配带引号的英文 "...,..."
            # 格式: 中文,"英文"
            match = re.match(r'^(.+),"(.*)"$', line)
            if match:
                zh = match.group(1).strip()
                en = match.group(2).strip()
            else:
                # 方法2: 找到最后一个逗号分隔
                # 从右边找最后一个逗号
                last_comma_pos = line.rfind(',')
                if last_comma_pos > 0:
                    zh = line[:last_comma_pos].strip()
                    en = line[last_comma_pos+1:].strip()
                else:
                    error_count += 1
                    continue
            
            # 跳过空值
            if not zh or not en:
                error_count += 1
                continue
            
            # 处理中文：原数据用空格分词，合并为连续文本供BPE重新分词
            zh = zh.replace(' ', '')
            
            # 跳过包含索引数字的中文（如 "0,1" 这样格式错误的行）
            if not zh or zh.isdigit():
                continue
            
            # 处理英文：清理HTML实体
            en = en.replace('&apos;', "'")
            en = en.replace('&quot;', '"')
            en = en.replace('&amp;', '&')
            en = en.replace('&lt;', '<')
            en = en.replace('&gt;', '>')
            en = en.replace('&nbsp;', ' ')
            en = en.replace('& nbsp', ' ')
            # 清理 &#数字; 格式的实体
            en = re.sub(r'&#\d+;', '', en)
            # 修复被错误拆分的 & amp ; -> & 或空格
            en = re.sub(r'&\s*amp\s*;?', ' ', en)
            en = re.sub(r'&\s*lt\s*;?', '<', en)
            en = re.sub(r'&\s*gt\s*;?', '>', en)
            en = en.replace('@-@', '')
            # 规范化空格
            en = ' '.join(en.split())
            
            if zh and en and len(zh) > 1 and len(en) > 1:
                zh_texts.append(zh)
                en_texts.append(en)
            else:
                error_count += 1
    
    # 确保对齐
    min_len = min(len(zh_texts), len(en_texts))
    zh_texts = zh_texts[:min_len]
    en_texts = en_texts[:min_len]
    
    print(f"有效句对: {min_len}")
    print(f"解析错误/跳过: {error_count}")
    
    if min_len == 0:
        print("错误：没有有效数据")
        return
    
    # 写入文件
    print(f"Writing to {output_zh} and {output_en}...")
    with open(output_zh, 'w', encoding='utf-8') as f:
        for text in zh_texts:
            f.write(text + '\n')
    
    with open(output_en, 'w', encoding='utf-8') as f:
        for text in en_texts:
            f.write(text + '\n')
    
    print(f"完成! 共 {min_len} 个句对")
    
    # 显示样本
    print("\n=== 样本预览 ===")
    for i in range(min(3, min_len)):
        print(f"中文: {zh_texts[i]}")
        print(f"英文: {en_texts[i]}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./data/WMT-CN-to-EN/wmt_zh_en_training_corpus_small.csv")
    parser.add_argument("--output_dir", type=str, default="./data/wmt_processed")
    parser.add_argument("--sample", type=int, default=None, help="Limit number of samples")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    basename = os.path.basename(args.input).replace('.csv', '')
    output_zh = os.path.join(args.output_dir, f"{basename}.zh")
    output_en = os.path.join(args.output_dir, f"{basename}.en")
    
    convert_wmt_data(args.input, output_zh, output_en, args.sample)


if __name__ == "__main__":
    main()