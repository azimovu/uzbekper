#!/usr/bin/env python3
"""Score WER + PER locally (VPS, CPU) from pod transcripts, with per-source breakdown.

PER uses uzg2p G2P on ref/hyp words. Foreign/empty tokens excluded.
Outputs both micro (primary) and per-source macro rows.
"""
import argparse, json, re, sys, os

UZG2P = os.path.join(os.path.dirname(__file__), '..', 'uzg2p')
sys.path.insert(0, UZG2P)

def normalize_text(t):
    t = t.lower()
    t = re.sub(r'[ʻʼ‘’\u02bb\u02bc]', '\u02bb', t)
    t = re.sub(r'[^a-z\u02bb\s]', ' ', t)
    return t.split()

def wer(refs, hyps):
    import editdistance
    tot_e = tot_n = 0
    for r, h in zip(refs, hyps):
        rw, hw = normalize_text(r), normalize_text(h)
        if not rw:
            continue
        tot_e += editdistance.eval(rw, hw)
        tot_n += len(rw)
    return 100 * tot_e / max(tot_n, 1)

def per(refs, hyps, g2p_cache, g_instance):
    import editdistance
    tot_e = tot_n = 0
    for r, h in zip(refs, hyps):
        rp = phonemize_words(normalize_text(r), g2p_cache, g_instance)
        hp = phonemize_words(normalize_text(h), g2p_cache, g_instance)
        rs = '#'.join(rp)
        hs = '#'.join(hp)
        if not rs:
            continue
        tot_e += editdistance.eval(list(rs), list(hs))
        tot_n += len(rs)
    return 100 * tot_e / max(tot_n, 1)

def phonemize_words(words, g2p_cache, g_instance):
    out = []
    for w in words:
        if w not in g2p_cache:
            try:
                g2p_cache[w] = g_instance(w).phonemes
            except Exception:
                g2p_cache[w] = ''
        p = g2p_cache[w]
        if p:
            out.append(p)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('transcripts', nargs='+')
    ap.add_argument('--out', default=None)
    ap.add_argument('--matched', action='store_true',
                    help='If 3+ transcript files given, score only shared clips (fair comparison)')
    args = ap.parse_args()

    from uzg2p import G2P
    g = G2P()

    if args.matched:
        label_map = {}
        for path in args.transcripts:
            label = os.path.basename(path).replace('transcripts_', '').replace('.json', '')
            label_map[label] = path
        results = score_matched(label_map, g)
        if args.out:
            json.dump(results, open(args.out, 'w'), indent=2, ensure_ascii=False)
            print(f'saved {args.out}')
        return

    results = {}
    for path in args.transcripts:
        data = json.load(open(path))
        model = os.path.basename(path).replace('transcripts_', '').replace('.json', '')
        cache = {}
        # micro
        w = wer([d['ref'] for d in data], [d['hyp'] for d in data])
        p = per([d['ref'] for d in data], [d['hyp'] for d in data], cache, g)
        # per-source
        sources = sorted({d.get('source', '') for d in data})
        per_src = {}
        for s in sources:
            sub = [d for d in data if d.get('source', '') == s]
            if not sub:
                continue
            sw = wer([d['ref'] for d in sub], [d['hyp'] for d in sub])
            sp = per([d['ref'] for d in sub], [d['hyp'] for d in sub], cache, g)
            per_src[s] = {'WER': round(sw, 2), 'PER': round(sp, 2),
                          'gap': round(sw - sp, 2), 'n': len(sub)}
        macro_wer = sum(v['WER'] for v in per_src.values()) / len(per_src)
        macro_per = sum(v['PER'] for v in per_src.values()) / len(per_src)
        results[model] = {
            'micro': {'WER': round(w, 2), 'PER': round(p, 2),
                      'gap': round(w - p, 2), 'n': len(data)},
            'macro': {'WER': round(macro_wer, 2), 'PER': round(macro_per, 2),
                      'gap': round(macro_wer - macro_per, 2)},
            'per_source': per_src,
        }
        print(f"{model}: micro WER {w:.2f}% PER {p:.2f}% gap {w-p:.2f} | "
              f"macro WER {macro_wer:.2f}% PER {macro_per:.2f}%")
        for s, v in per_src.items():
            print(f"    {s}: WER {v['WER']:.2f} PER {v['PER']:.2f} n={v['n']}")

    if args.out:
        json.dump(results, open(args.out, 'w'), indent=2, ensure_ascii=False)
        print(f'saved {args.out}')

def score_matched(label_map, g):
    """Score multiple models on clips they share (matched by ref text) for fair comparison."""
    all_data = {}
    for label, path in label_map.items():
        all_data[label] = json.load(open(path))

    ref_sets = [set(r['ref'] for r in all_data[l]) for l in label_map]
    shared = set.intersection(*ref_sets) if ref_sets else set()
    print(f"Shared clips across all models: {len(shared)}")

    results = {}
    for label in label_map:
        data = [r for r in all_data[label] if r['ref'] in shared]
        cache = {}
        w = wer([d['ref'] for d in data], [d['hyp'] for d in data])
        p = per([d['ref'] for d in data], [d['hyp'] for d in data], cache, g)
        sources = sorted({d.get('source', '') for d in data})
        per_src = {}
        for s in sources:
            sub = [d for d in data if d.get('source', '') == s]
            if not sub: continue
            sw = wer([d['ref'] for d in sub], [d['hyp'] for d in sub])
            sp = per([d['ref'] for d in sub], [d['hyp'] for d in sub], cache, g)
            per_src[s] = {'WER': round(sw, 2), 'PER': round(sp, 2),
                          'gap': round(sw - sp, 2), 'n': len(sub)}
        results[label] = {
            'micro': {'WER': round(w, 2), 'PER': round(p, 2),
                      'gap': round(w - p, 2), 'n': len(data)},
            'per_source': per_src,
        }
        print(f"{label}: micro WER {w:.2f}% PER {p:.2f}% gap {w-p:.2f} | n={len(data)}")
        for s, v in per_src.items():
            print(f"    {s}: WER {v['WER']:.2f} PER {v['PER']:.2f} n={v['n']}")
    return results

if __name__ == '__main__':
    main()
