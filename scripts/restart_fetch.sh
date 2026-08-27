#!/bin/bash
# Restart pod audio fetch with HF token installed
set -e
INFO=/home/ubuntu/.config/vastai/uzbekper_instance.json
HOST=$(python3 -c "import json;print(json.load(open('$INFO'))['ssh_host'])")
PORT=$(python3 -c "import json;print(json.load(open('$INFO'))['ssh_port'])")
TOKEN=$(grep "^HF_TOKEN=" ~/.openclaw/.env | cut -d= -f2)
SSH="ssh -i $HOME/.ssh/vast_uzbek_tts -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT root@$HOST"

$SSH "
pkill -f fetch_test_audio || true; sleep 1
echo '$TOKEN' > /workspace/hf_token
mkdir -p ~/.cache/huggingface && echo '$TOKEN' > ~/.cache/huggingface/token
echo token-installed: \$(cut -c1-10 /workspace/hf_token)...
cd /workspace/uzbekper
setsid bash -c 'python3 fetch_test_audio.py --joins-dir data --manifest data/test_manifest.json --out-root ./audios > /workspace/logs/fetch.log 2>&1; echo FETCH-DONE >> /workspace/logs/fetch.log' < /dev/null > /dev/null 2>&1 &
sleep 8
tail -3 /workspace/logs/fetch.log
"
