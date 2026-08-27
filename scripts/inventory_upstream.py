"""Inventory upstream source repos: file layout + manifest/parquet structure.

Repos (from uzinfocom 701h README sources table):
  common_voice -> mozilla-foundation/common_voice_17_0 (config uz) or yakhyo mirror
  uzbekvoice   -> DavronSherbaev/uzbekvoice-filtered / ai4uz/uzbekvoice-filtered
  usc          -> issai/Uzbek_Speech_Corpus or murodbek/uzbek-speech-corpus
  news_youtube -> islomov/news_youtube_uzbek_speech_dataset
  it_youtube   -> islomov/it_youtube_uzbek_speech_dataset
  podcasts     -> islomov/podcasts_tashkent_dialect_youtube_uzbek_speech_dataset
"""
import json, urllib.request, urllib.parse
H = {'User-Agent': 'bench'}

def api(u):
    return json.load(urllib.request.urlopen(urllib.request.Request(u, headers=H)))

repos = [
    'islomov/news_youtube_uzbek_speech_dataset',
    'islomov/it_youtube_uzbek_speech_dataset',
    'islomov/podcasts_tashkent_dialect_youtube_uzbek_speech_dataset',
    'murodbek/uzbek-speech-corpus',
    'DavronSherbaev/uzbekvoice-filtered',
    'issai/Uzbek_Speech_Corpus',
]
for r in repos:
    try:
        d = api(f'https://huggingface.co/api/datasets/{r}/tree/main?limit=50')
        print(f'\n== {r} ==')
        for x in d[:15]:
            sz = x.get('size', 0)
            print(f"  {x['path']}  ({sz/1e6:.1f} MB)" if sz else f"  {x['path']}/")
    except Exception as e:
        print(r, 'ERR', e)
