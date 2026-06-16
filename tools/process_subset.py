"""
处理 wmt_zh_en_training_corpus_subset.csv
这个数据集已经清洗过，中文有空格分词，需要：
1. 去掉中文的空格（重新合并为连续文本）
2. 清理英文的HTML实体
"""

import re
import os

def convert_subset_data(csv_file, output_zh, output_en):
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
            match = re.match(r'^(.+),"(.*)"$', line)
            if match:
                zh = match.group(1).strip()
                en = match.group(2).strip()
            else:
                # 尝试找到最后一个逗号分隔
                last_comma_pos = line.rfind(',')
                if last_comma_pos > 0:
                    zh = line[:last_comma_pos].strip()
                    en = line[last_comma_pos+1:].strip()
                else:
                    error_count += 1
                    continue
            
            # 去掉中文中的空格（原数据已分词）
            zh = zh.replace(' ', '')
            
            # 跳过空值
            if not zh or not en:
                error_count += 1
                continue
            
            # 处理英文：清理HTML实体
            en = en.replace('&apos;', "'")
            en = en.replace('&quot;', '"')
            en = en.replace('&amp;', '&')
            en = en.replace('&lt;', '<')
            en = en.replace('&gt;', '>')
            en = en.replace('&nbsp;', ' ')
            en = re.sub(r'&#\d+;', '', en)
            en = re.sub(r'&\s*amp\s*;?', ' ', en)
            en = re.sub(r'@\s*-\s*@', '', en)  # 处理 @-@
            en = ' '.join(en.split())
            
            if zh and en and len(zh) > 1 and len(en) > 1:
                zh_texts.append(zh)
                en_texts.append(en)
            else:
                error_count += 1
    
    min_len = min(len(zh_texts), len(en_texts))
    zh_texts = zh_texts[:min_len]
    en_texts = en_texts[:min_len]
    
    print(f"有效句对: {min_len}")
    print(f"解析错误/跳过: {error_count}")
    
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
        print(f"中文: {zh_texts[i][:50]}...")
        print(f"英文: {en_texts[i][:50]}...")
        print()


if __name__ == "__main__":
    input_file = "./data/WMT-CN-to-EN/wmt_zh_en_training_corpus_subset.csv"
    output_dir = "./data/wmt_processed"
    os.makedirs(output_dir, exist_ok=True)
    
    output_zh = os.path.join(output_dir, "subset.zh")
    output_en = os.path.join(output_dir, "subset.en")
    
    convert_subset_data(input_file, output_zh, output_en)