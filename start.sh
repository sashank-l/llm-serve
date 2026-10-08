#!/bin/bash
set -e
cd /home/surya/Downloads/llm-serve
source venv/bin/activate
pkill -f run_server.py 2>/dev/null || true
sleep 1
nohup python3 run_server.py --model /home/surya/models/qwen2.5-1.5b --device cuda --port 8000 \
  > /home/surya/Downloads/llm-serve/server.log 2>&1 &
echo "Server PID: $!"
echo "Server started in background. Logs: /home/surya/Downloads/llm-serve/server.log"
