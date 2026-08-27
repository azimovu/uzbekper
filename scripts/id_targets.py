"""ID-based join for uzbekvoice / common_voice / usc sources.
Our stems: uzbekvoice = '597360' (numeric); usc = 'usc_638880527_1_16345_1';
common_voice = 'common_voice_validated_38034'.
Probe upstream id/path fields for matching tokens."""
import json, re, collections

rows = [json.loads(l) for l in open('/home/ubuntu/uzbekper/data/test_manifest.json')]
by_src = collections.defaultdict(list)
for r in rows:
    stem = r['audio_filepath'].split('/')[-1].rsplit('.', 1)[0]
    by_src[r['source']].append({'stem': stem, 'key': re.sub(r"[^a-z']+", ' ', r['text'].lower())})

for src, items in by_src.items():
    print(src, len(items), items[0]['stem'])
json.dump({k: v for k, v in by_src.items()},
          open('/home/ubuntu/uzbekper/data/id_join_targets.json', 'w'), ensure_ascii=False)
