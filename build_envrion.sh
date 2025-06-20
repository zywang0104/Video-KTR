#!/bin/bash

printf "\n" | sudo pip3 uninstall vllm
printf "\n" | pip3 uninstall vllm
bash setup.sh

cd src/qwen-vl-utils
pip install -e .[decord]
cd ../..

cd /mnt/bn/tns-live-mllm/private/wangzy/Video-R1/
pip3 install vllm==0.7.2

printf "\n" | sudo pip3 uninstall transformers
printf "\n" | pip3 uninstall transformers

cd /opt/tiger/haggsX/
hdfs dfs -get hdfs://harunava/user/ziyue.wang/transformers-main_.zip
unzip transformers-main_.zip
cd transformers-main
python3 -m pip install .

cd /mnt/bn/tns-live-mllm/private/wangzy/Video-R1/
python3 login.py
pip3 show transformers
python3 -c "import transformers; print(transformers.__version__, transformers.__file__)"
pip3 show trl

# python3 download.py

# bash /mnt/bn/tns-live-mllm/private/wangzy/Video-R1/src/scripts/run_grpo_video.sh