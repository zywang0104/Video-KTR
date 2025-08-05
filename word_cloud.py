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
file_path = '/Users/bytedance/Desktop/WangZy/videoR1_dev/video-R1-Live/combo_tokens.txt'

# 读取文本
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# text = replace_partial_word(text, 'person', ratio=1)
# text = replace_partial_word(text, 'image', ratio=1)
# text = replace_partial_word(text, 'cube', ratio=1)
# text = replace_partial_word(text, 'cylinder', ratio=1)
# text = replace_partial_word(text, 'triangle', ratio=1)
# tokens = re.findall(r'\w+|[^\w\s]', text)  # 匹配单词或单字符标点/符号
# tokens = [
#     token for token in tokens 
#     if not (token.isdigit() or token in ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
# ]
# word_freq = Counter(tokens)

text = replace_partial_word(text,'person',ratio=0.5)
text = replace_partial_word(text,'video',ratio=0.5)
text = replace_partial_word(text,'First',ratio=0.5)
text = replace_partial_word(text,'now',ratio=0.3)

# 生成词云
wc = WordCloud(width=800, height=600,colormap='Set2',background_color='white').generate(text)

# unselected token
# wc = WordCloud(
#     width=800,
#     height=600,
#     background_color='white',
#     colormap='Greys',
#     collocations=False,         # 不合并常见词对
#     normalize_plurals=False,    # 不统一复数形式
#     regexp=None,                # 禁用内部分词正则
#     stopwords=None              # 不使用默认停用词
# ).generate_from_frequencies(word_freq)

# 显示词云
plt.figure(figsize=(15, 7.5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.savefig("wordcloudSelected.pdf", format='pdf')  # 或 .eps
plt.show()
