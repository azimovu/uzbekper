"""ID join for murodbek/uzbek-speech-corpus: our stems 'usc_638880527_1_16345_1'
vs upstream id '1000201291_1_3686_4' (same composite format, no 'usc_' prefix)."""
import json, re

def norm(t):
    t = (t or '').lower()
    t = t.replace('\u02bb', "'").replace('\u02bc', "'").replace('\u2018', "'").replace('\u2019', "'")
    t = re.sub(r"[^a-z']+", ' ', t)
    return ' '.join(t.split())

rows = [json.loads(l) for l in open('/home/ubuntu/uzbekper/data/test_manifest.json')]
by_id, by_text = {}, {}
for i, r in enumerate(rows):
    if r['source'] != 'usc':
        continue
    stem = r['audio_filepath'].split('/')[-1].rsplit('.', 1)[0]
    iid = stem[4:] if stem.startswith('usc_') else stem
    by_id[iid] = {'idx': i, 'matched': False}
    by_text[norm(r['text'])] = i
print(f'{len(by_id)} usc targets', flush=True)

from datasets import load_dataset
ds = load_dataset('murodbek/uzbek-speech-corpus', split='train', streaming=True)
seen = m_id = m_text = 0
for batch in ds.with_format('arrow'):
    ids = batch.column('id').to_pylist()
    texts = batch.column('sentence').to_pylist()
    seen += len(ids)
    for rid, t in zip(ids, texts):
        if rid in by_id and not by_id[rid]['matched']:
            by_id[rid]['matched'] = True
            m_id += 1
        k = norm(t)
        if k in by_text:
            j = by_text[k]
            if not by_id.get(j, {}).get('matched'):
                # text-only match for rows whose ID didn't match
                for v in by_id.values():
                    if v['idx'] == j and not v['matched']:
                        v['matched'] = True
                        v['via'] = 'text'
                        m_text += 1
                        break
    if seen % 20000 == 0:
        print(f'seen {seen}, by_id {m_id}, via_text {m_text}', flush=True)

n = sum(1 for v in by_id.values() if v['matched'])
print(f'\nRESULT [usc]: {n}/{len(by_id)} ({100*n/len(by_id):.1f}%) after {seen}')
json.dump({'scanned': seen, 'matched': n, 'total': len(by_id), 'rows': by_id},
          open('/home/ubuntu/uzbekper/data/join_usc.json', 'w'), ensure_ascii=False)
