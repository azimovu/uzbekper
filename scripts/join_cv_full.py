"""CV join, full-download per split instead of streaming (streaming flaked)."""
import json, re

def norm(t):
    t = (t or '').lower()
    for a in ['\u02bb', '\u02bc', '\u2018', '\u2019']:
        t = t.replace(a, "'")
    t = re.sub(r"[^a-z']+", ' ', t)
    return ' '.join(t.split())

rows = [json.loads(l) for l in open('data/test_manifest.json')]
by_num = {}; by_text = {}
for i, r in enumerate(rows):
    if r['source'] != 'common_voice':
        continue
    stem = r['audio_filepath'].split('/')[-1].rsplit('.', 1)[0]
    m = re.search(r'(\d+)$', stem)
    num = m.group(1) if m else stem
    by_num[num] = {'idx': i, 'matched': False}
    by_text[norm(r['text'])] = i
print(len(by_num), 'targets', flush=True)
idx_matched = {}  # idx -> True once matched (duplicate normalized texts share an idx)

from datasets import load_dataset
total = 0
for split in ['train', 'validation', 'test', 'other']:
    try:
        ds = load_dataset('yakhyo/mozilla-common-voice-uzbek', split=split)
    except Exception as e:
        print(split, 'skip', type(e).__name__, flush=True)
        continue
    for batch in ds.with_format('arrow'):
        texts = batch.column('sentence').to_pylist()
        total += len(texts)
        for t in texts:
            k = norm(t)
            j = by_text.get(k)
            if j is not None and not idx_matched.get(j):
                # mark all by_num rows with this idx (unique per idx)
                for num, v in by_num.items():
                    if v['idx'] == j and not v['matched']:
                        v['matched'] = True
                        v['split'] = split
                        idx_matched[j] = True
                        break
    nm = sum(1 for v in by_num.values() if v['matched'])
    print(f'after {split}: {nm}/{len(by_num)} (scanned {total})', flush=True)

n = sum(1 for v in by_num.values() if v['matched'])
print(f'RESULT [common_voice]: {n}/{len(by_num)} ({100*n/len(by_num):.1f}%) scanned {total}')
json.dump({'matched': n, 'total': len(by_num), 'rows': by_num},
          open('data/join_common_voice.json', 'w'), ensure_ascii=False)
