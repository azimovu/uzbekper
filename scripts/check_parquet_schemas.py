"""Check parquet schemas of the data/ dirs: do they carry audio columns?"""
import json, urllib.request
H = {'User-Agent': 'bench'}

def api(u):
    return json.load(urllib.request.urlopen(urllib.request.Request(u, headers=H)))

repos = [
    'islomov/news_youtube_uzbek_speech_dataset',
    'murodbek/uzbek-speech-corpus',
    'DavronSherbaev/uzbekvoice-filtered',
]
for r in repos:
    try:
        d = api(f'https://huggingface.co/api/datasets/{r}/tree/main/data?limit=10')
        print(f'\n== {r} ==')
        for x in d[:6]:
            print(f"  {x['path']}  ({x.get('size',0)/1e6:.1f} MB)")
        # schema via datasets-server first rows
        import urllib.parse
        u = ('https://datasets-server.huggingface.co/first-rows?dataset='
             + urllib.parse.quote(r, safe='') + '&config=default&split=train')
        fr = api(u)
        feats = fr.get('features', [])
        print('  features:', [(f['name'], f['type'].get('_type') or f['type']) for f in feats])
    except Exception as e:
        print(r, 'ERR', e)
