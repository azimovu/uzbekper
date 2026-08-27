"""Pilot join for a given source (parametrized). Usage:
   join_pilot.py <upstream_repo> <source_tag> <out.json>"""
import json, re, sys

repo, src, out = sys.argv[1], sys.argv[2], sys.argv[3]

def norm(t):
    t = t.lower()
    t = t.replace('\u02bb', "'").replace('\u02bc', "'").replace('\u2018', "'").replace('\u2019', "'")
    t = re.sub(r"[^a-z']+", ' ', t)
    return ' '.join(t.split())

rows = [json.loads(l) for l in open('/home/ubuntu/uzbekper/data/test_manifest.json')]
targets = {}
for i, r in enumerate(rows):
    if r['source'] == src:
        targets[norm(r['text'])] = {'idx': i, 'dur': r['duration'], 'matched': False,
                                    'raw': r['text']}
print(f'{len(targets)} target texts for {src}', flush=True)

from datasets import load_dataset
ds = load_dataset(repo, split='train', streaming=True)
seen = matched = 0
for batch in ds.with_format('arrow'):
    cols = batch.column_names
    idcol = 'id' if 'id' in cols else ('path' if 'path' in cols else cols[0])
    txtcol = 'text' if 'text' in cols else 'sentence'
    ids = batch.column(idcol).to_pylist()
    texts = batch.column(txtcol).to_pylist()
    seen += len(ids)
    for rid, t in zip(ids, texts):
        k = norm(t or '')
        if k in targets and not targets[k]['matched']:
            targets[k]['matched'] = True
            targets[k]['up_id'] = rid
            matched += 1
    if seen % 5000 == 0:
        print(f'seen {seen}, matched {matched}', flush=True)

n = sum(1 for v in targets.values() if v['matched'])
print(f'\nRESULT [{src}]: {n}/{len(targets)} matched ({100*n/len(targets):.1f}%) '
      f'after scanning {seen} upstream rows')
json.dump({'repo': repo, 'scanned': seen, 'matched': n, 'total': len(targets),
           'rows': targets}, open(out, 'w'), ensure_ascii=False)
