#!/bin/bash
# Cleanup pod disk: delete large-v3 cache + dataset caches, restart fetch
INFO=/home/ubuntu/.config/vastai/uzbekper_instance.json
HOST=$(python3 -c "import json;print(json.load(open('$INFO'))['ssh_host'])")
PORT=$(python3 -c "import json;print(json.load(open('$INFO'))['ssh_port'])")
SSH="ssh -i $HOME/.ssh/vast_uzbek_tts -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -p $PORT root@$HOST"
$SSH 'pkill -9 -f "fetch_test_audio" 2>/dev/null; pkill -9 -f "benchmark_infer" 2>/dev/null; sleep 1; rm -rf /root/.cache/huggingface/hub/models--openai--whisper-large-v3 /root/.cache/huggingface/hub/datasets--* /root/.cache/pip; df -h / | tail -1 | awk "{print \"disk:\", \$4, \"free\"}"'
echo "--- restarting fetch ---"
$SSH 'cd /workspace/uzbekper && nohup python3 fetch_test_audio.py --joins-dir data --manifest data/test_manifest.json --out-root ./audios --sources news_youtube,it_youtube,podcasts_dialect > /workspace/logs/fetch4.log 2>&1 & sleep 10; tail -2 /workspace/logs/fetch4.log'
