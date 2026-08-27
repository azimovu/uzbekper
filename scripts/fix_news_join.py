"""Normalize join_news_youtube.json to the standard shape {rows:{key:{...}}}
so fetch_test_audio.py's load_targets() reads all six files identically."""
import json

d = json.load(open('/home/ubuntu/uzbekper/data/join_news_youtube.json'))
# old pilot wrote the targets dict directly (text-keyed), no 'rows' wrapper
assert 'rows' not in d, 'already wrapped'
matched = sum(1 for v in d.values() if isinstance(v, dict) and v.get('matched'))
json.dump({'repo': 'islomov/news_youtube_uzbek_speech_dataset',
           'scanned': 20795, 'matched': matched, 'total': len(d),
           'rows': d},
          open('/home/ubuntu/uzbekper/data/join_news_youtube.json', 'w'),
          ensure_ascii=False)
print(f'wrapped: total {len(d)}, matched {matched}')
