#!/bin/bash
# Relaunch pod audio fetch (token now installed in cache)
INFO=/home/ubuntu/.config/vastai/uzbekper_instance.json
HOST=$(python3 -c "import json;print(json.load(open('$INFO'))['ssh_host'])")
PORT=$(python3 -c "import json;print(json.load(open('$INFO'))['ssh_port'])")
SSH="ssh -i $HOME/.ssh/vast_uzbek_tts -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -p $PORT root@$HOST"
$SSH 'pkill -f fetch_test_audio || true; sleep 1; cd /workspace/uzbekper; setsid bash -c "python3 fetch_test_audio.py --joins-dir data --manifest data/test_manifest.json --out-root ./audios > /workspace/logs/fetch.log 2>&1; echo FETCH-DONE >> /workspace/logs/fetch.log" < /dev/null > /dev/null 2>&1 & sleep 10; tail -3 /workspace/logs/fetch.log'
