"""Join common_voice source. Our stems: 'common_voice_train_22429' /
'common_voice_validated_38034'. Upstream: mozilla-foundation/common_voice_17_0
config uz has client_id/path/sentence columns where path like
'common_voice_uz_38034.mp3'. Try both train+validation+test splits of CV17,
fall back to yakhyo mirror if gated."""
import json, re

def norm(t):
    t = (t or '').lower()
    t = t.replace('\u02bb', "'").replace('\u02bc', "'").replace('\u2018', "'").replace('\u2019', "'")
    t = re.sub(r"[^a-z']+", ' ', t)
    return ' '.join(t.split())

rows = [json.loads(l) for l in open('/home/ubuntu/uzbekper/data/test_manifest.json')]
by_num, by_text = {}, {}
for i, r in enumerate(rows):
    if r['source'] != 'common_voice':
        continue
    stem = r['audio_filepath'].split('/')[-1].rsplit('.', 1)[0]
    m = re.search(r'(\d+)$', stem)
    num = m.group(1) if m else stem
    by_num[num] = {'idx': i, 'stem': stem, 'matched': False}
    by_text[norm(r['text'])] = i
print(f'{len(by_num)} cv targets', flush=True)

from datasets import load_dataset, get_dataset_config_names
total_seen = 0
# CV17 official is script-gated; use yakhyo mirror (no path column -> text join only)
for split in ['train', 'validation', 'test', 'other']:
    try:
        ds = load_dataset('yakhyo/mozilla-common-voice-uzbek',
                          split=split, streaming=True)
    except Exception as e:
        print(f'{split}: skip ({type(e).__name__})', flush=True)
        continue
    m_id = sum(1 for v in by_num.values() if v['matched'])
    print(f'-- {split} open', flush=True)
    try:
        seen_local = 0
        for batch in ds.with_format('arrow'):
            cols = batch.column_names
            paths = batch.column('path').to_pylist() if 'path' in cols else [''] * batch.num_rows
            texts = batch.column('sentence').to_pylist() if 'sentence' in cols else batch.column('text').to_pylist()
            total_seen += len(paths)
            seen_local += len(paths)
            for p, t in zip(paths, texts):
                m = re.search(r'(\d+)', str(p))
                if m and m.group(1) in by_num:
                    v = by_num[m.group(1)]
                    if not v['matched']:
                        v['matched'] = True
                        v['split'] = split
                elif t is not None:
                    k = norm(t)
                    if k in by_text:
                        j = by_text[k]
                        v2 = next(v for v in by_num.values() if v['idx'] == j)
                        if not v2['matched']:
                            v2['matched'] = True
                            v2['via'] = 'text'
                            v2['split'] = split
            nm = sum(1 for v in by_num.values() if v['matched'])
            if seen_local % 20000 < len(paths):
                print(f'seen {total_seen}, matched {nm}', flush=True)
            if nm == len(by_num):
                break
    except StopIteration:
        print(f'{split}: stream ended (empty or truncated)', flush=True)
        continue
    nm = sum(1 for v in by_num.values() if v['matched'])
    print(f'after {split}: {nm}/{len(by_num)}', flush=True)

n = sum(1 for v in by_num.values() if v['matched'])
print(f'\nRESULT [common_voice]: {n}/{len(by_num)} ({100*n/len(by_num):.1f}%) after {total_seen}')
json.dump({'scanned': total_seen, 'matched': n, 'total': len(by_num), 'rows': by_num},
          open('/home/ubuntu/uzbekper/data/join_common_voice.json', 'w'), ensure_ascii=False)
