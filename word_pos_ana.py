import spacy
from collections import Counter
# 如果尚未安装 wordcloud 库，请先在终端中运行:
# pip install wordcloud

from wordcloud import WordCloud
from collections import Counter
import matplotlib.pyplot as plt
import re
import random

import re
import random

def replace_partial_word(text, word, ratio=0.5):
    """
    随机删除 text 中部分指定单词（默认删除一半）
    
    参数：
        text (str): 输入文本
        word (str): 要删除的目标单词（区分大小写，受 IGNORECASE 控制）
        ratio (float): 删除比例，范围 0~1

    返回：
        str: 处理后的文本
    """
    # 找出所有匹配的单词位置（忽略大小写）
    pattern = rf'\b{re.escape(word)}\b'
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    total = len(matches)
    
    if total == 0 or ratio <= 0:
        return text  # 无匹配或不删除
    
    k = min(int(total * ratio), total)
    to_replace = random.sample(matches, k=k)

    # 转换成字符列表，按位置逆序删除
    text_list = list(text)
    for match in sorted(to_replace, key=lambda m: m.start(), reverse=True):
        start, end = match.span()
        text_list[start:end] = ''  # 删除匹配词

    return ''.join(text_list)


# 替换为您的文件路径
file_path = '/Users/bytedance/Desktop/WangZy/videoR1_dev/video-R1-Live/unselected_tokens.txt'

# 读取文本
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()
text = replace_partial_word(text, 'person', ratio=1)
text = replace_partial_word(text, 'vehicles', ratio=1)
text = replace_partial_word(text, 'think', ratio=1)
text = replace_partial_word(text, 'answer', ratio=1)

text = replace_partial_word(text, 'image', ratio=1)
text = replace_partial_word(text, 'cube', ratio=1)
text = replace_partial_word(text, 'cylinder', ratio=1)
text = replace_partial_word(text, 'triangle', ratio=1)
# 加载模型
nlp = spacy.load("en_core_web_sm")

doc = nlp(text)

# 统计词性
pos_counts = Counter(token.pos_ for token in doc if token.is_alpha)  # 排除标点

# 计算总数和比例
total = sum(pos_counts.values())
pos_ratio = {pos: f"{(count / total):.2%}" for pos, count in pos_counts.items()}

print("词性比例:", pos_ratio)