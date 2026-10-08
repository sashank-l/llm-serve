#!/usr/bin/env python3
"""
run_server.py
Entry point to start the Mini LLM Inference Engine server.

Usage:
  python run_server.py --model ~/models/qwen2.5-1.5b --port 8000 --device cuda
"""
import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Mini LLM Inference Server")
    parser.add_argument("--model",   required=True,          help="Path to model directory")
    parser.add_argument("--host",    default="0.0.0.0",      help="Bind host")
    parser.add_argument("--port",    default=8000, type=int, help="Bind port")
    parser.add_argument("--device",  default="cuda",         help="torch device: cuda or cpu")
    parser.add_argument("--workers", default=1,   type=int,  help="Uvicorn workers (1 recommended)")
    args = parser.parse_args()

    # Import here so the module-level init_engine can be called before uvicorn starts
    from mini_engine.server.app import init_engine
    init_engine(args.model, args.device)

    uvicorn.run(
        "mini_engine.server.app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
