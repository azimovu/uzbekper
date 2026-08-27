"""Dry-run: verify load_targets + path resolution without downloading."""
import sys, json, os
sys.path.insert(0, 'scripts')
src = open('scripts/fetch_test_audio.py').read()
ns = {}
exec(compile(src.split('def iter_parquet_batches')[0], 'x', 'exec'), ns)
targets = ns['load_targets']('data')
manifest = [json.loads(l) for l in open('data/test_manifest.json')]
mbi = {i: r for i, r in enumerate(manifest)}
total = 0
for s, m in targets.items():
    n = 0
    for k, v in m.items():
        if v.get('matched'):
            row = mbi[v['idx']]
            assert 'audio_filepath' in row
            n += 1
    total += n
    print(f'{s}: {n} resolvable')
print('TOTAL:', total)
