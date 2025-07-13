import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True, help='Model name to search results for')
args = parser.parse_args()
file = args.model_name
# HDFS 源目录
HDFS_DIR = f'hdfs://harunava/user/ziyue.wang/videor1_models/{file}/'
LOCAL_DIR = f'/mnt/bn/tns-live-mllm/private/wangzy/Video-R1/holmes_models/{file}'
os.makedirs(LOCAL_DIR,exist_ok=True)
# 并发线程数，根据机器和网络能力调整
MAX_WORKERS = 500

def list_hdfs_files(hdfs_dir):
    """
    调用 `hdfs dfs -ls -R` 列出 hdfs_dir 下所有文件，并返回文件路径列表
    """
    cmd = ['hdfs', 'dfs', '-ls', '-R', hdfs_dir]
    output = subprocess.check_output(cmd, text=True)
    files = []
    for line in output.splitlines():
        parts = line.split()
        # ls -R 的文件行通常以权限开头，例如 "-rw-r--r--"
        if len(parts) >= 8 and parts[0].startswith('-'):
            # 最后一列是完整的 HDFS 路径
            files.append(parts[-1])
    return files

def download_from_hdfs(hdfs_path):
    """
    下载单个文件 hdfs_path 到本地对应位置，保留相对结构
    """
    # 计算相对路径
    rel = os.path.relpath(hdfs_path, HDFS_DIR)
    local_dest = os.path.join(LOCAL_DIR, rel)
    local_parent = os.path.dirname(local_dest)

    # 确保本地目录存在
    os.makedirs(local_parent, exist_ok=True)

    # 下载并覆盖
    subprocess.run(['hdfs', 'dfs', '-get', '-f', hdfs_path, local_dest], check=True)
    return rel

def main():
    # 1. 列出所有 HDFS 文件
    print(f'正在列出 HDFS 上 {HDFS_DIR} 下的所有文件…')
    hdfs_files = list_hdfs_files(HDFS_DIR)
    total = len(hdfs_files)
    print(f'共发现 {total} 个文件，使用 {MAX_WORKERS} 个线程并行下载。')

    # 2. 并发下载
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_path = { pool.submit(download_from_hdfs, path): path for path in hdfs_files }
        for fut in tqdm(as_completed(future_to_path), total=total, desc='下载进度'):
            path = future_to_path[fut]
            try:
                rel = fut.result()
                tqdm.write(f'[OK]   下载完成: {rel}')
            except subprocess.CalledProcessError as e:
                tqdm.write(f'[ERR]  下载失败: {path}，命令输出：{e}')

if __name__ == '__main__':
    main()
