import json, urllib.request, urllib.parse

key = open('/home/ubuntu/.config/vastai/vast_api_key').read().strip()

def search(gpu_name):
    q = {'gpu_name': {'eq': gpu_name}, 'num_gpus': {'eq': 1},
         'inet_up': {'gte': 100}, 'disk_space': {'gte': 80},
         'rentable': {'eq': True}, 'order': [('dph_total', 'asc')]}
    url = f'https://console.vast.ai/api/v0/bundles/?{urllib.parse.urlencode({"q": json.dumps(q)})}'
    req = urllib.request.Request(url, headers={'Authorization': f'Bearer {key}'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r).get('offers', [])

best = {}
for name in ['RTX 4090', 'RTX 5090']:
    offers = search(name)
    print(f'\n=== {name}: {len(offers)} offers ===')
    for o in offers[:5]:
        print(f"  ${o['dph_total']:.2f}/hr | VRAM {o.get('gpu_ram',0)/1024:.0f}GB | "
              f"RAM {o['cpu_ram']/1024:.0f}GB | up:{o.get('inet_up',0):.0f}Mbps | "
              f"disk {o.get('disk_space',0):.0f}GB | reliab {(o.get('reliability2') or 0)*100:.0f}% | "
              f"cores {o.get('cpu_cores_effective',0):.0f}")
    solid = [o for o in offers
             if (o.get('reliability2') or 0) >= 0.95
             and o.get('cpu_ram', 0)/1024 >= 30
             and o.get('inet_up', 0) >= 200]
    if solid:
        b = solid[0]
        best[name] = {
            'offer_id': b['id'],
            'price_hr': round(b['dph_total'], 2),
            'vram_gb': round(b.get('gpu_ram', 0)/1024),
            'ram_gb': round(b['cpu_ram']/1024),
            'uplink_mbps': round(b.get('inet_up', 0)),
            'disk_gb': round(b.get('disk_space', 0)),
            'reliability': f"{(b.get('reliability2') or 0)*100:.0f}%",
            'cpu_cores': int(b.get('cpu_cores_effective', 0)),
        }

print('\n=== RECOMMENDED (reliab>=95%, sysRAM>=30GB, uplink>=200Mbps) ===')
print(json.dumps(best, indent=1))
with open('/home/ubuntu/uzbek-tts/vast_candidates.json', 'w') as f:
    json.dump(best, f, indent=1)
