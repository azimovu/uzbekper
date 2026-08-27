"""Pilot: stream islomov/it_youtube_uzbek_speech_dataset (id+text only),
build normalized-text index, match against our test_manifest news_youtube rows.
Streaming avoids downloading audio column bytes."""
import json, re, sys

def norm(t):
    t = t.lower()
    t = t.replace('\u02bb', "'").replace('\u02bc', "'").replace('\u2018', "'").replace('\u2019', "'")
    t = re.sub(r"[^a-z']+", ' ', t)
    return ' '.join(t.split())

rows = [json.loads(l) for l in open('/home/ubuntu/uzbekper/data/test_manifest.json')]
targets = {}
for i, r in enumerate(rows):
    if r['source'] it_youtube:
        targets[norm(r['text'])] = {'idx': i, 'dur': r['duration'], 'matched': False}
print(f'{len(targets)} target texts', flush=True)

from datasets import load_dataset
ds = load_dataset('islomov/it_youtube_uzbek_speech_dataset', split='train',
                  streaming=True)
seen = matched = 0
for batch in ds.with_format('arrow'):
    ids = batch.column('id').to_pylist()
    texts = batch.column('text').to_pylist()
    seen += len(ids)
    for rid, t in zip(ids, texts):
        k = norm(t)
        if k in targets:
            targets[k]['matched'] = True
            targets[k]['up_id'] = rid
            matched += 1
    if seen % 2000 == 0:
        print(f'seen {seen}, matched {matched}', flush=True)

n = sum(1 for v in targets.values() if v['matched'])
print(f'\nRESULT: {n}/{len(targets)} news_youtube test rows matched '
      f'({100*n/len(targets):.1f}%) after scanning {seen} upstream rows')
json.dump(targets, open('/home/ubuntu/uzbekper/data/join_it_youtube.json', 'w'),
          ensure_ascii=False)
