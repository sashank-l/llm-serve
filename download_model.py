#!/usr/bin/env python3
"""
download_model.py
Download a HuggingFace model and tokenizer to a local directory.
Run this on the remote machine before starting the server.

Usage:
  python download_model.py --model Qwen/Qwen2.5-1.5B-Instruct --dir ~/models/qwen2.5-1.5b
"""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace model")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HuggingFace model repo ID")
    parser.add_argument("--dir",   default=os.path.expanduser("~/models/qwen2.5-1.5b"),
                        help="Local download directory")
    parser.add_argument("--token", default=None, help="HuggingFace access token (if needed)")
    args = parser.parse_args()

    dest = os.path.expanduser(args.dir)
    os.makedirs(dest, exist_ok=True)

    print(f"Downloading {args.model!r} → {dest}")
    from huggingface_hub import snapshot_download
    path = snapshot_download(
        repo_id        = args.model,
        local_dir      = dest,
        token          = args.token,
        ignore_patterns= ["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )
    print(f"Done. Model saved to: {path}")


if __name__ == "__main__":
    main()
