#!/usr/bin/env python3
"""Language pin verification: run 5 clips through whisper with explicit language='uz',
check for Turkish chars. Must pass before trusting any whisper run."""
import argparse, json, re, os, sys

BASE = '/home/ubuntu/uzbekper'
turk = re.compile(r'[üİıŞĞÇ]')

def load_manifest(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def normalize_text(t):
    t = t.lower()
    t = re.sub(r'[ʻʼ‘’\u02bb\u02bc]', '\u02bb', t)
    t = re.sub(r'[^a-z\u02bb\s]', ' ', t)
    return t.split()

def check_hypotheses(model_id, rows, audio_root, language='uz', n_test=5):
    """Run n_test clips through whisper and check for Turkish chars."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    import librosa, numpy as np

    proc = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16).cuda()
    model.eval()

    forced_ids = proc.get_decoder_prompt_ids(language=language, task='transcribe')

    print(f"=== Language pin verification: {model_id} ===")
    print(f"forced_decoder_ids: {forced_ids}")

    turk_count = 0
    for i, r in enumerate(rows[:n_test]):
        path = os.path.join(audio_root, r['audio_filepath'])
        if not os.path.exists(path):
            print(f"  [{i}] audio missing: {path}")
            continue
        a, _ = librosa.load(path, sr=16000)
        if len(a) < 1600:
            a = np.pad(a, (0, 1600 - len(a)))
        feat = proc(a, sampling_rate=16000, return_tensors='np').input_features[0]
        inp = torch.from_numpy(feat).half().cuda().unsqueeze(0)
        with torch.no_grad():
            ids = model.generate(inp, forced_decoder_ids=forced_ids, max_new_tokens=256)
        hyp = proc.batch_decode(ids, skip_special_tokens=True)[0]
        has_turk = bool(turk.search(hyp))
        if has_turk:
            turk_count += 1
        print(f"  [{i}] ref={repr(r['text'][:60])}")
        print(f"       hyp={repr(hyp[:80])} turk={'YES' if has_turk else 'no'}")

    print(f"\nResult: {turk_count}/{n_test} hyps have Turkish chars")
    if turk_count > 0:
        print("FAIL: Language pin is NOT working. Do not trust this model's output.")
        return False
    else:
        print("PASS: Language pin verified — output is clean Uzbek.")
        return True

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['v3', 'gitnazarov', 'blueraccoon'])
    ap.add_argument('--manifest', default='data/test_manifest.json')
    ap.add_argument('--audio-root', default='./audios')
    ap.add_argument('--n', type=int, default=5)
    args = ap.parse_args()

    model_map = {
        'v3': 'openai/whisper-large-v3',
        'gitnazarov': 'GitNazarov/whisper-large-uz',
        'blueraccoon': 'BlueRaccoon/whisper-small-uz',
    }

    rows = load_manifest(args.manifest)
    ok = check_hypotheses(model_map[args.model], rows, args.audio_root, n_test=args.n)
    sys.exit(0 if ok else 1)
