#!/usr/bin/env python3
"""Materialize matched test audio on the pod (fetch_test_audio.py).

Reads join outputs (data/join_*.json copied to pod) + test_manifest.json,
downloads ONLY the 9,320 matched clips from upstream HF repos, writes
audios/<original relative path> so manifest paths resolve unchanged.

Strategy per source:
  uzbekvoice  - DavronSherbaev/uzbekvoice-filtered parquet, ID join -> stream rows,
                save by original stem. Uses datasets streaming + pyarrow.
  usc         - murodbek/uzbek-speech-corpus parquet, ID+text join
  common_voice- yakhyo/mozilla-common-voice-uzbek parquet, text join
  news/it/pod - islomov/* parquets, text join

All repos embed audio in parquet; we iterate shards, extract matched rows only.
Resumable: skips clips whose target file already exists with size >1KB.

Usage: python3 fetch_test_audio.py --joins-dir data --manifest data/test_manifest.json \
           --out-root ./audios [--sources uzbekvoice,usc]
"""
import argparse, json, os, re, sys, hashlib

def norm(t):
    t = (t or '').lower()
    for a in ['\u02bb', '\u02bc', '\u2018', '\u2019']:
        t = t.replace(a, "'")
    t = re.sub(r"[^a-z']+", ' ', t)
    return ' '.join(t.split())

def load_targets(joins_dir):
    """Return {source: {upstream_key: manifest_row}} from join outputs.

    Delegates to uzbekper_pipeline.fetch_targets (unit-tested; the old inline
    version was fine here, but counting elsewhere double-counted fields).
    """
    import sys as _sys
    _pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _pkg_root not in _sys.path:
        _sys.path.insert(0, _pkg_root)
    from uzbekper_pipeline.fetch_targets import load_targets as _load

    return _load(joins_dir)

def stem_of(relpath):
    return os.path.basename(relpath).rsplit('.', 1)[0]

def save_wav(audio_bytes_or_struct, out_path):
    """datasets Audio feature decodes to {'bytes':..., 'path':...} in arrow format."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    b = audio_bytes_or_struct.get('bytes') if isinstance(audio_bytes_or_struct, dict) \
        else audio_bytes_or_struct
    if not b or len(b) < 1024:
        return False
    with open(out_path, 'wb') as f:
        f.write(b)
    return True

def iter_parquet_batches(repo, split='train'):
    """Stream arrow batches (id/text/audio columns) without full decode."""
    from datasets import load_dataset
    ds = load_dataset(repo, split=split, streaming=True)
    for batch in ds.with_format('arrow'):
        yield batch

def find_audio_col(cols):
    for c in ('audio', 'path'):
        if c in cols:
            return c
    raise KeyError(f'no audio column in {cols}')

def run_source(src, repo, targets, text_index, manifest_by_idx, out_root,
               split='train', stats=None):
    """targets: matched-key dict; text_index: norm_text -> idx (for text joins);
    manifest_by_idx: idx -> manifest row (for output path)."""
    from datasets import load_dataset
    need = sum(len(v) for v in targets.get(src, {}).values())
    print(f'[{src}] {need} clips to fetch from {repo}', flush=True)
    for split_name in ([split] if isinstance(split, str) else split):
        if stats['done'] >= need:
            break
        try:
            ds = load_dataset(repo, split=split_name, streaming=True)
        except Exception as e:
            print(f'[{src}] split {split_name} skip: {e}', flush=True)
            continue
        seen_local = 0
        try:
            for batch in ds.with_format('arrow'):
                cols = batch.column_names
                acol = find_audio_col(cols)
                texts = (batch.column('sentence').to_pylist()
                         if 'sentence' in cols else
                         batch.column('text').to_pylist() if 'text' in cols
                         else [None] * batch.num_rows)
                ids = (batch.column('id').to_pylist()
                       if 'id' in cols else
                       batch.column('path').to_pylist() if 'path' in cols
                       else [''] * batch.num_rows)
                audios = batch.column(acol).to_pylist()
                seen_local += batch.num_rows
                for rid, t, aud in zip(ids, texts, audios):
                    if stats['done'] >= need:
                        break
                    row_ref = None
                    # ID-based match first
                    base = str(rid).split('/')[-1].rsplit('.', 1)[0]
                    alt = base[4:] if base.startswith('usc_') else base
                    for cand in (base, alt):
                        if cand in targets.get(src, {}):
                            v = targets[src][cand]
                            if not v.get('_fetched'):
                                row_ref = v
                            break
                    # text-based fallback
                    if row_ref is None and t is not None:
                        k = norm(t)
                        if k in targets.get(src, {}):
                            v = targets[src][k]
                            if not v.get('_fetched'):
                                row_ref = v
                    if row_ref is None or row_ref.get('_fetched'):
                        continue
                    idx = row_ref['idx']
                    relpath = manifest_by_idx[idx]['audio_filepath']
                    out_path = os.path.join(out_root, relpath)
                    if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
                        row_ref['_fetched'] = True
                        stats['done'] += 1
                        continue
                    if save_wav(aud, out_path):
                        row_ref['_fetched'] = True
                        stats['done'] += 1
                        if stats['done'] % 250 == 0:
                            print(f'[{src}] {stats["done"]}/{need} fetched '
                                  f'(scanned {seen_local})', flush=True)
                if stats['done'] >= need:
                    break
        except Exception as e:
            print(f'[{src}] stream error at {seen_local}: {type(e).__name__} '
                  f'{str(e)[:120]} — continuing', flush=True)
        print(f'[{src}] split {split_name}: {stats["done"]}/{need}', flush=True)

REPOS = {
    'uzbekvoice': ('DavronSherbaev/uzbekvoice-filtered',
                   ['train', 'validation', 'test', 'other']),
    'usc': ('murodbek/uzbek-speech-corpus', ['train', 'test']),
    'common_voice': ('yakhyo/mozilla-common-voice-uzbek',
                     ['train', 'validation', 'test', 'other']),
    'news_youtube': ('islomov/news_youtube_uzbek_speech_dataset', ['train']),
    'it_youtube': ('islomov/it_youtube_uzbek_speech_dataset', ['train']),
    'podcasts_dialect': ('islomov/podcasts_tashkent_dialect_youtube_'
                         'uzbek_speech_dataset', ['train']),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--joins-dir', default='data')
    ap.add_argument('--manifest', default='data/test_manifest.json')
    ap.add_argument('--out-root', default='./audios')
    ap.add_argument('--sources', default=None, help='comma list; default all')
    args = ap.parse_args()

    manifest = [json.loads(l) for l in open(args.manifest)]
    manifest_by_idx = {i: r for i, r in enumerate(manifest)}
    targets = load_targets(args.joins_dir)

    # build text index: normalized ref text -> idx (for text-joined sources)
    text_index = {norm(r['text']): i for i, r in enumerate(manifest)}

    wanted = args.sources.split(',') if args.sources else list(REPOS)
    # manifest source tag -> join-file name (only podcasts differs)
    _alias = {'podcasts_dialect': 'podcasts'}
    stats = {'done': 0}
    total_needed = 0
    for s in wanted:
        tkey = _alias.get(s, s)
        if tkey in targets:
            targets[tkey] = {k: v for k, v in targets[tkey].items()
                             if not v.get('_fetched')}
            total_needed += len(targets[tkey])   # record count, not fields
    print(f'total to fetch: {total_needed}', flush=True)

    for s in wanted:
        if s not in REPOS:
            print(f'unknown source {s}, skip'); continue
        repo, splits = REPOS[s]
        stats['done'] = 0  # reset per source
        run_source(s, repo, targets, text_index, manifest_by_idx,
                   args.out_root, split=splits, stats=stats)

    print('\n=== SUMMARY ===')
    grand = 0
    for s in wanted:
        tkey = _alias.get(s, s)
        n = sum(1 for v in targets.get(tkey, {}).values() if v.get('_fetched'))
        grand += n
        print(f'{s}: {n}/{len(targets.get(tkey, {}))} fetched this run')
    print(f'TOTAL: {grand}/{total_needed}')
    # final verification: count files on disk vs manifest expectation
    ok = missing = 0
    for i, r in enumerate(manifest):
        p = os.path.join(args.out_root, r['audio_filepath'])
        if os.path.exists(p) and os.path.getsize(p) > 1024:
            ok += 1
        elif any(v.get('_fetched') for tgt in targets.values()
                 for v in tgt.values() if v.get('idx') == i):
            missing += 1
    print(f'on disk OK: {ok}, claimed-fetched-but-missing: {missing}')

if __name__ == '__main__':
    main()
