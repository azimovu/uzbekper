import json, urllib.request
H = {'User-Agent': 'b'}
# check remaining README of yakhyo for license mention
rd = urllib.request.urlopen(urllib.request.Request(
    'https://huggingface.co/datasets/yakhyo/mozilla-common-voice-uzbek/raw/main/README.md',
    headers=H)).read().decode()
print(rd[800:2000])
