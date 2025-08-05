import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import re

# 文件路径
with open('/Users/bytedance/Desktop/WangZy/videoR1_dev/video-R1-Live/high_dep_tokens.txt', 'r', encoding='utf-8') as f:
    text1 = f.read()
with open('/Users/bytedance/Desktop/WangZy/videoR1_dev/video-R1-Live/high_entropy_tokens.txt', 'r', encoding='utf-8') as f:
    text2 = f.read()
with open('/Users/bytedance/Desktop/WangZy/videoR1_dev/video-R1-Live/high_temp_dep_tokens.txt', 'r', encoding='utf-8') as f:
    text3 = f.read()

# 处理函数：保留前 n 个词
def preprocess(text, n=2000):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    return set(tokens[:n])

# 限制每个文本只取前 2000 个词
words1 = preprocess(text1, n=2000)
words2 = preprocess(text2, n=2000)
words3 = preprocess(text3, n=2000)

# 绘制 Venn 图
plt.figure(figsize=(8, 6))
venn = venn3([words1, words2, words3], ('Visual-Aware Tokens', 'Entropy-Aware Tokens', 'Temporal-Aware Tokens'))

for patch in venn.patches:
    if patch:
        patch.set_alpha(0.4)  # 透明度设置
# 百分比替换标签
total_words = len(words1 | words2 | words3)
for subset in ('100', '010', '001', '110', '101', '011', '111'):
    if venn.get_label_by_id(subset):
        count = venn.get_label_by_id(subset).get_text()
        if count is not None:
            count = int(count)
            percentage = f"{(count / total_words * 100):.1f}%"
            venn.get_label_by_id(subset).set_text(percentage)

plt.tight_layout()
plt.savefig("venn.pdf", format="pdf")
plt.show()
