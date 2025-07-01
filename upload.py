import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # 新增

# 本地待上传目录
LOCAL_DIR = '/opt/tiger/video-R1-Live/models'
# HDFS 目标目录
HDFS_DIR = 'hdfs://harunava/user/ziyue.wang/videor1_models'

# 并发线程数，根据你的机器和集群能力调整
MAX_WORKERS = 500

def list_all_files(root_dir):
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            files.append(os.path.join(dirpath, fn))
    return files

def upload_to_hdfs(local_path):
    rel_path = os.path.relpath(local_path, LOCAL_DIR)
    hdfs_dest = os.path.join(HDFS_DIR, rel_path).replace('\\', '/')
    hdfs_parent = os.path.dirname(hdfs_dest)

    # 确保目录存在
    subprocess.run(['hdfs', 'dfs', '-mkdir', '-p', hdfs_parent], check=True)
    # 上传并覆盖
    subprocess.run(['hdfs', 'dfs', '-put', '-f', local_path, hdfs_dest], check=True)
    return rel_path

def main():
    files = list_all_files(LOCAL_DIR)
    total = len(files)
    print(f'发现 {total} 个文件，使用 {MAX_WORKERS} 个线程并行上传。')

    # 提交所有任务
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_file = { pool.submit(upload_to_hdfs, f): f for f in files }
        
        # tqdm 会根据已完成的 future 自动更新进度并给出 ETA
        for fut in tqdm(as_completed(future_to_file), total=total, desc='上传进度'):
            local_file = future_to_file[fut]
            try:
                rel = fut.result()
                tqdm.write(f'[OK]   上传完成: {rel}')
            except subprocess.CalledProcessError as e:
                tqdm.write(f'[ERR]  上传失败: {local_file}，命令输出：{e}')

if __name__ == '__main__':
    main()
