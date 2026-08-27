"""Probe islomov news_youtube parquet: does its 'id' match our stems?
Read ONE parquet's id column remotely via HTTP range is complex; instead
download the smallest shard? All are ~450MB. Better: use datasets-server
rows API to sample rows and inspect 'id' values."""
import json, urllib.request, urllib.parse
H = {'User-Agent': 'bench'}

def api(u):
    return json.load(urllib.request.urlopen(urllib.request.Request(u, headers=H)))

for r in ['islomov/news_youtube_uzbek_speech_dataset',
          'islomov/it_youtube_uzbek_speech_dataset',
          'islomov/podcasts_tashkent_dialect_youtube_uzbek_speech_dataset',
          'murodbek/uzbek-speech-corpus']:
    u = (f'https://datasets-server.huggingface.co/rows?dataset='
         + urllib.parse.quote(r, safe='') + '&config=default&split=train&offset=0&length=3')
    try:
        d = api(u)
        for row in d['rows'][:3]:
            rid = row['row'].get('id')
            txt = row['row'].get('text') or row['row'].get('sentence')
            aud = row['row'].get('audio') or row['row'].get('path')
            print(f"{r.split('/')[0]} | id={rid!r} | text={str(txt)[:60]!r} | audio_keys={list(aud.keys()) if isinstance(aud,dict) else aud}")
    except Exception as e:
        print(r, 'ERR', e)
