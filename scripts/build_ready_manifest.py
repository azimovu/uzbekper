#!/usr/bin/env python3
"""Build filtered manifest of clips whose audio already exists on disk."""
import json, os, sys

manifest_path = sys.argv[1] if len(sys.argv) > 1 else 'data/test_manifest.json'
audio_root = sys.argv[2] if len(sys.argv) > 2 else './audios'
out = sys.argv[3] if len(sys.argv) > 3 else 'data/ready_manifest.json'

rows = [json.loads(l) for l in open(manifest_path)]
ready = []
for r in rows:
    p = os.path.join(audio_root, r['audio_filepath'])
    if os.path.exists(p) and os.path.getsize(p) > 1024:
        ready.append(r)
with open(out, 'w') as f:
    for r in ready:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'{len(ready)}/{len(rows)} ready -> {out}')
