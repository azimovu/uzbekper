"""Build join keys from OUR test manifest: normalized text + source tag.
Output: per-source counts to know what we must match where.
Also probe whether upstream 'id' fields can match our audio_filepath stems
(e.g. news_youtube_1635 -> id 1635?)."""
import json, re, collections

def norm(t):
    t = t.lower()
    t = t.replace('\u02bb', "'").replace('\u02bc', "'").replace('\u2018', "'").replace('\u2019', "'")
    t = re.sub(r'[^a-z\']+', ' ', t)
    return ' '.join(t.split())

rows = [json.loads(l) for l in open('/home/ubuntu/uzbekper/data/test_manifest.json')]
by_src = collections.defaultdict(list)
for i, r in enumerate(rows):
    stem = r['audio_filepath'].split('/')[-1].rsplit('.', 1)[0]
    by_src[r['source']].append({'idx': i, 'path': r['audio_filepath'], 'stem': stem,
                                'text': r['text'], 'key': norm(r['text']),
                                'dur': r['duration']})

for src, items in sorted(by_src.items()):
    stems = [x['stem'] for x in items[:5]]
    print(f'{src}: {len(items)} rows | sample stems: {stems}')

# check id-extractability per source
import re as _re
for src in ['news_youtube', 'it_youtube', 'podcasts_dialect']:
    items = by_src[src]
    ids = [_re.search(r'(\d+)$', x['stem']) for x in items]
    n = sum(1 for m in ids if m)
    print(f'{src}: {n}/{len(items)} stems end with digits')
