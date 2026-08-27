#!/bin/bash
# Quick pod status: fetch progress, files, disk, inference processes
INFO=/home/ubuntu/.config/vastai/uzbekper_instance.json
HOST=$(python3 -c "import json;print(json.load(open('$INFO'))['ssh_host'])")
PORT=$(python3 -c "import json;print(json.load(open('$INFO'))['ssh_port'])")
timeout 60 ssh -i $HOME/.ssh/vast_uzbek_tts -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -p $PORT root@$HOST '
echo "fetch: $(grep -E "fetched|split" /workspace/logs/fetch2.log | tail -1)"
echo "files: $(find /workspace/uzbekper/audios -name "*.wav" | wc -l)"
echo "disk:  $(df -h / | tail -1 | awk "{print \$4}")"
echo "infer procs: $(pgrep -f benchmark_infer | wc -l)"
' 2>/dev/null
