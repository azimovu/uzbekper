#!/usr/bin/env python3
"""Pod-side ASR inference for UzbekPER benchmark (v2 — incremental push).

Runs ON a Vast GPU pod. Downloads test audio, runs each model zero-shot,
checkpoints every N utterances and pushes JSONL chunks to the VPS immediately.

Artifacts (canonical, consumed by benchmark_score.py):
  transcripts_<safe_model>.json  - full JSON array
  chunks/<safe_model>/chunk_<start>_<end>.jsonl - incremental push units

Each record: clip_id, idx, ref, hyp, source, model, revision, decode, ts.
clip_id = audio_filepath (stable across manifest changes); idx = position (display only).
RESUME KEYED BY clip_id, NOT by list position — manifest growth no longer breaks resume.

Usage:
  python3 benchmark_infer.py --manifest test_manifest.json \
      --audio-root ./audios --models nvidia_fastconformer,whisper_large_v3 \
      --push-target user@vps:/path/out [--push-key /path/key] [--limit N]

PROVENANCE: --audio-root must hold wavs materialized by fetch_test_audio.py.
"""
import argparse, json, os, subprocess, sys, time

def load_manifest(path, limit=None):
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            r['idx'] = i          # position in current manifest (for display only)
            r['clip_id'] = r.get('audio_filepath', f'clip_{i}')  # STABLE ID for resume
            rows.append(r)
    if limit:
        rows = rows[:limit]
    return rows

def safe(s):
    return s.replace('/', '_')

def push_chunk(chunk_path, target, key=None):
    """Fire-and-forget scp of one chunk; failures are logged, never fatal."""
    cmd = ['scp', '-o', 'StrictHostKeyChecking=no', '-o', 'BatchMode=yes']
    if key:
        cmd += ['-i', key]
    cmd += [chunk_path, f'{target}/']
    try:
        subprocess.run(cmd, check=True, timeout=120,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        print(f'[push FAIL] {chunk_path}: {e}', flush=True)
        return False

class Recorder:
    """Accumulates results; writes chunk files + final merged JSON."""
    def __init__(self, model_id, revision, decode, outdir='.', chunk_every=200,
                 push_target=None, push_key=None):
        self.model_id, self.revision, self.decode = model_id, revision, decode
        self.chunk_every = chunk_every
        self.push_target, self.push_key = push_target, push_key
        self.records = []
        self.chunk_start = 0
        self.cdir = os.path.join(outdir, 'chunks', safe(model_id))
        os.makedirs(self.cdir, exist_ok=True)

    def add(self, clip_id, idx, ref, hyp, source):
        self.records.append({'clip_id': clip_id, 'idx': idx, 'ref': ref, 'hyp': hyp,
                             'source': source, 'model': self.model_id,
                             'revision': self.revision, 'decode': self.decode,
                             'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                                 time.gmtime())})
        n = len(self.records) - self.chunk_start
        if n >= self.chunk_every:
            self._flush()

    def _flush(self):
        end = len(self.records)
        path = os.path.join(self.cdir,
                            f'chunk_{self.chunk_start:05d}_{end:05d}.jsonl')
        with open(path, 'w') as f:
            for r in self.records[self.chunk_start:end]:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f'[checkpoint] {path} ({end - self.chunk_start} recs)', flush=True)
        if self.push_target:
            ok = push_chunk(path, self.push_target, self.push_key)
            print(f'[push] {"ok" if ok else "FAILED (will retry at next flush)"}',
                  flush=True)
        self.chunk_start = end

    def finalize(self, outdir='.'):
        if len(self.records) > self.chunk_start:
            self._flush()
        # merged full file from local records (authoritative copy also on pod)
        path = os.path.join(outdir, f'transcripts_{safe(self.model_id)}.json')
        json.dump(self.records, open(path, 'w'), ensure_ascii=False)
        print(f'wrote {path} ({len(self.records)} records)', flush=True)
        # push the merged file too, as belt-and-braces
        if self.push_target:
            push_chunk(path, self.push_target, self.push_key)

def run_nemo(model_id, rows, audio_root, rec):
    import nemo.collections.asr as nemo_asr
    asr = nemo_asr.models.ASRModel.from_pretrained(model_id)
    t0 = time.time()
    for i, r in enumerate(rows):
        path = os.path.join(audio_root, r['audio_filepath'])
        try:
            hyp = asr.transcribe([path])[0][0]
        except Exception as e:
            hyp = f'<ERROR:{type(e).__name__}>'
        rec.add(r['clip_id'], i, r['text'], hyp, r.get('source', ''))
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f'{model_id}: {i+1}/{len(rows)} ({el:.0f}s, '
                  f'{el/(i+1):.2f}s/utt)', flush=True)
    return rec

def run_whisper(model_id, rows, audio_root, rec, language='uz', batch_size=16,
                verify_pin=True):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    import librosa, numpy as np
    proc = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16).cuda()
    model.eval()
    forced_ids = proc.get_decoder_prompt_ids(language=language, task='transcribe')

    if verify_pin and rows:
        print(f'[lang-pin] verifying on first {min(5, len(rows))} clips...', flush=True)
        turk_re = __import__('re').compile(r'[üİıŞĞÇ]')
        bad = 0
        for r in rows[:min(5, len(rows))]:
            path = os.path.join(audio_root, r['audio_filepath'])
            if not os.path.exists(path):
                continue
            try:
                a, _ = librosa.load(path, sr=16000)
                if len(a) < 1600:
                    a = np.pad(a, (0, 1600 - len(a)))
            except Exception:
                continue
            feat = proc(a, sampling_rate=16000, return_tensors='np').input_features[0]
            inp = torch.from_numpy(feat).half().cuda().unsqueeze(0)
            with torch.no_grad():
                ids = model.generate(inp, forced_decoder_ids=forced_ids,
                                     max_new_tokens=256)
            hyp = proc.batch_decode(ids, skip_special_tokens=True)[0]
            if turk_re.search(hyp):
                bad += 1
                print(f'[lang-pin] FAIL clip {r["audio_filepath"]}: {repr(hyp[:80])}', flush=True)
        if bad > 0:
            print(f'[lang-pin] {bad}/5 hyps have Turkish chars — ABORTING. Language pin not holding.', flush=True)
            raise SystemExit(1)
        print('[lang-pin] PASS — language pin verified, proceeding.', flush=True)
    t0 = time.time()
    B = batch_size
    for start in range(0, len(rows), B):
        batch_rows = rows[start:start + B]
        audios = []
        for r in batch_rows:
            path = os.path.join(audio_root, r['audio_filepath'])
            try:
                a, _ = librosa.load(path, sr=16000)
                if len(a) < 1600:  # pad sub-0.1s
                    import numpy as np
                    a = np.pad(a, (0, 1600 - len(a)))
            except Exception:
                a = None
            audios.append(a)
        valid_idx = [j for j, a in enumerate(audios) if a is not None]
        hyps = ['<ERROR:LoadError>'] * len(batch_rows)
        if valid_idx:
            feats = [proc(a, sampling_rate=16000, return_tensors='np'
                          ).input_features[0] for a in
                     (audios[j] for j in valid_idx)]
            import torch as T
            inp = T.stack([T.from_numpy(f) for f in feats]).half().cuda()
            with T.no_grad():
                ids = model.generate(inp, forced_decoder_ids=forced_ids,
                                     max_new_tokens=256)
            decoded = proc.batch_decode(ids, skip_special_tokens=True)
            for j, h in zip(valid_idx, decoded):
                hyps[j] = h
        for j, r in enumerate(batch_rows):
            rec.add(r['clip_id'], start + j, r['text'], hyps[j], r.get('source', ''))
        el = time.time() - t0
        done_n = min(start + B, len(rows))
        print(f'{model_id}: {done_n}/{len(rows)} ({el:.0f}s, '
              f'{el/done_n:.2f}s/utt, bs={B})', flush=True)
    return rec

MODELS = {
    'nvidia_fastconformer': ('nemo',
        'nvidia/stt_uz_fastconformer_hybrid_large_pc'),
    'uzinfocom_fastconformer': ('nemo',
        'uzinfocom-edu-ai/asr-uz-fastconformer-large'),
    'whisper_large_v3': ('whisper', 'openai/whisper-large-v3'),
    'whisper_small_uz': ('whisper', 'BlueRaccoon/whisper-small-uz'),
    'whisper_large_uz': ('whisper', 'GitNazarov/whisper-large-uz'),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--audio-root', required=True)
    ap.add_argument('--models', default='nvidia_fastconformer')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--chunk-every', type=int, default=200)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--no-verify-pin', action='store_true',
                    help='Skip 5-clip language-pin verification before inference')
    ap.add_argument('--push-target', default=None,
                    help='user@host:/dest/dir for incremental scp')
    ap.add_argument('--push-key', default=None, help='ssh key path')
    args = ap.parse_args()

    rows = load_manifest(args.manifest, args.limit)
    decode_cfg = {'decoding': 'greedy/default', 'batch': args.batch_size,
                  'language': 'uz'}

    for key in args.models.split(','):
        kind, mid = MODELS[key.strip()]
        rev = mid  # TODO: pin exact HF revision hash before the rental run
        # resume: drop clips already present in prior transcripts/chunks (by STABLE clip_id)
        done_ids = set()
        merged_path = f'transcripts_{safe(mid)}.json'
        chunk_glob = os.path.join('chunks', safe(mid), '*.jsonl')
        import glob as _g
        for p in [merged_path] + sorted(_g.glob(chunk_glob)):
            try:
                if p.endswith('.jsonl'):
                    for line in open(p):
                        d = json.loads(line)
                        done_ids.add(d.get('clip_id', d.get('idx')))
                else:
                    for d in json.load(open(p)):
                        done_ids.add(d.get('clip_id', d.get('idx')))
            except Exception:
                pass
        todo = [r for r in rows if r['clip_id'] not in done_ids]
        print(f'=== {mid} === {len(todo)} to run ({len(done_ids)} already done)',
              flush=True)
        if not todo:
            continue
        rec = Recorder(mid, rev, decode_cfg, chunk_every=args.chunk_every,
                       push_target=args.push_target, push_key=args.push_key)
        fn = run_nemo if kind == 'nemo' else run_whisper
        fn(mid, todo, args.audio_root, rec, language='uz',
             batch_size=args.batch_size, verify_pin=not args.no_verify_pin)
        rec.finalize()
        # merge with previous records into one authoritative file (by STABLE clip_id)
        allrecs = {}
        for d in json.load(open(merged_path)):
            allrecs[d.get('clip_id', d['idx'])] = d
        for p in sorted(_g.glob(chunk_glob)):
            try:
                for line in open(p):
                    d = json.loads(line)
                    allrecs[d.get('clip_id', d['idx'])] = d
            except Exception:
                pass
        json.dump(sorted(allrecs.values(), key=lambda x: x.get('idx', 0)),
                  open(merged_path, 'w'), ensure_ascii=False)
        print(f'merged total for {mid}: {len(allrecs)}', flush=True)

if __name__ == '__main__':
    main()
