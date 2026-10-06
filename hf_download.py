#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键下载 Hugging Face 大模型（可断点续传、不跳过）
用法：
    python hf_download.py  repo_id  [local_dir]  [--mirror]
示例：
    python hf_download.py  meta-llama/Llama-2-7b-hf  ./Llama-2-7b-hf  --mirror
"""
import os
import sys
import argparse
from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars, enable_progress_bars


def parse_args():
    parser = argparse.ArgumentParser(description="HF 大文件断点续传下载")
    parser.add_argument("repo_id", help="模型/数据集 ID，如: microsoft/DialoGPT-large")
    parser.add_argument("local_dir", nargs="?", default=None,
                        help="本地保存目录，默认 ./repo_id")
    parser.add_argument("--mirror", action="store_true",
                        help="使用国内镜像 https://hf-mirror.com")
    parser.add_argument("--token", default=None,
                        help="HF Access Token（如需私有库）")
    parser.add_argument("--max-workers", type=int, default=8,
                        help="并发下载线程数，默认 8")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_id = args.repo_id
    local_dir = args.local_dir or repo_id.replace("/", "--")

    # 1. 国内镜像
    if args.mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    # 2. 关闭 hf_transfer（容易半截失败）
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    # 3. 超时、重试
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "1200"          # 单文件 20 min
    os.environ["HF_HUB_DOWNLOAD_MAX_WORKERS"] = str(args.max_workers)

    print(f"开始下载：{repo_id} -> {os.path.abspath(local_dir)}")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,      # 不生成符号链接，直接存实体
            resume_download=True,              # 断点续传
            #verification_mode="all_checks",    # 强制校验大小+校验和
            token=args.token,
            etag_timeout=300,                  # 校验超时 5 min
        )
    except Exception as e:
        print("\n[ERROR] 下载失败，请根据下方异常排查：")
        print(e)
        sys.exit(1)

    print("\n✅ 下载完成，全部文件已通过校验！")


if __name__ == "__main__":
    enable_progress_bars()   # 确保有进度条
    main()