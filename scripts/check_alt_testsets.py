import json, urllib.request
H = {'User-Agent': 'bench'}
d = json.loads(urllib.request.urlopen(urllib.request.Request(
    'https://datasets-server.huggingface.co/size?dataset=google%2Ffleurs&config=uz_uz',
    headers=H)).read())
for s in d['size']['splits']:
    print('fleurs uz_uz', s['split'], s['num_rows'])

rows = [json.loads(l) for l in open('/home/ubuntu/uzbekper/data/test_manifest.json')]
print('our test split:', len(rows), 'utts,', round(sum(r['duration'] for r in rows) / 3600, 1), 'h')
import collections
by = collections.defaultdict(float)
n = collections.Counter()
for r in rows:
    by[r['source']] += r['duration']
    n[r['source']] += 1
for k in sorted(by, key=lambda k: -by[k]):
    print(f'  {k}: {n[k]} utts, {by[k]/3600:.1f} h')
