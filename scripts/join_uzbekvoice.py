"""ID/text join for DavronSherbaev/uzbekvoice-filtered (has original_sentence_id,
path, text columns). Stream id+text only."""
import json, re, sys

def norm(t):
    t = (t or '').lower()
    t = t.replace('\u02bb', "'").replace('\u02bc', "'").replace('\u2018', "'").replace('\u2019', "'")
    t = re.sub(r"[^a-z']+", ' ', t)
    return ' '.join(t.split())

rows = [json.loads(l) for l in open('/home/ubuntu/uzbekper/data/test_manifest.json')]
targets_by_id, targets_by_text = {}, {}
for i, r in enumerate(rows):
    if r['source'] != 'uzbekvoice':
        continue
    stem = r['audio_filepath'].split('/')[-1].rsplit('.', 1)[0]
    targets_by_id[stem] = {'idx': i, 'matched': False}
    targets_by_text[norm(r['text'])] = i
print(f'{len(targets_by_id)} uzbekvoice targets', flush=True)

from datasets import load_dataset
ds = load_dataset('DavronSherbaev/uzbekvoice-filtered', split='train', streaming=True)
seen = m_id = m_text = 0
for batch in ds.with_format('arrow'):
    cols = batch.column_names
    ids = batch.column('id').to_pylist() if 'id' in cols else [''] * batch.num_rows
    paths = batch.column('path').to_pylist() if 'path' in cols else [''] * batch.num_rows
    orig = batch.column('original_sentence_id').to_pylist() if 'original_sentence_id' in cols else [''] * batch.num_rows
    texts = batch.column('text').to_pylist()
    seen += len(ids)
    for rid, p, oid, t in zip(ids, paths, orig, texts):
        # our stems look like 597360 — check any field equal to it
        for cand in (str(rid), str(p), str(oid)):
            base = str(cand).split('/')[-1].rsplit('.', 1)[0]
            if base in targets_by_id and not targets_by_id[base]['matched']:
                targets_by_id[base]['matched'] = True
                targets_by_id[base]['up_field'] = cand[:40]
                m_id += 1
        k = norm(t)
        if k in targets_by_text:
            j = targets_by_text[k]
            if not any(v['idx'] == j and v['matched'] for v in targets_by_id.values()):
                m_text += 1
    if seen % 20000 == 0:
        print(f'seen {seen}, by_id {m_id}, text_extra {m_text}', flush=True)

n = sum(1 for v in targets_by_id.values() if v['matched'])
print(f'\nRESULT [uzbekvoice]: by_id {n}/{len(targets_by_id)} ({100*n/len(targets_by_id):.1f}%) after {seen}')
json.dump({'scanned': seen, 'by_id': n, 'total': len(targets_by_id),
           'rows': targets_by_id}, open('/home/ubuntu/uzbekper/data/join_uzbekvoice.json', 'w'),
          ensure_ascii=False)
