"""Rebuild join_common_voice.json keyed by normalized text (fetcher expects
text keys for text-join sources; the CV join was number-keyed)."""
import json, re

d = json.load(open('/home/ubuntu/uzbekper/data/join_common_voice.json'))
manifest = [json.loads(l) for l in open('/home/ubuntu/uzbekper/data/test_manifest.json')]

def norm(t):
    t = (t or '').lower()
    for a in ['\u02bb', '\u02bc', '\u2018', '\u2019']:
        t = t.replace(a, "'")
    return ' '.join(re.sub(r"[^a-z']+", ' ', t).split())

new_rows = {}
for k, v in d['rows'].items():
    if v.get('matched'):
        txt = norm(manifest[v['idx']]['text'])
        new_rows[txt] = {'idx': v['idx'], 'matched': True}
json.dump({'repo': 'yakhyo/mozilla-common-voice-uzbek',
           'scanned': d.get('scanned', 200935),
           'matched': len(new_rows), 'total': len(new_rows), 'rows': new_rows},
          open('/home/ubuntu/uzbekper/data/join_common_voice.json', 'w'),
          ensure_ascii=False)
print('rewritten with', len(new_rows), 'text-keyed entries')
