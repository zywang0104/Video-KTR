# 如果尚未安装 wordcloud 库，请先在终端中运行:
# pip install wordcloud

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 替换为您的文件路径
file_path = '/Users/bytedance/Desktop/WangZy/videoR1_dev/video-R1-Live/high_entropy_tokens_batch_level.txt'

# 读取文本
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# 生成词云
wc = WordCloud(width=800, height=400,colormap='magma',background_color='white').generate(text)

# 显示词云
plt.figure(figsize=(15, 7.5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
