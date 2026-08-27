"""Generate stratified 500-word validation set for Utkirbek's native-speaker review."""
import random
import csv

random.seed(42)

# Load full vocab with frequencies
vocab = []
with open('/home/ubuntu/uzbek-tts/g2p_audit/vocab_full.txt') as f:
    for line in f:
        w, c = line.rstrip('\n').split('\t')
        vocab.append((w, int(c)))

freq = dict(vocab)
words = [w for w, c in vocab]

def has(w, chars): return any(ch in w for ch in chars)
def is_ascii_apos(w): return "'" in w

strata = {k: [] for k in [
    'high_freq_native',   # top frequency, no apostrophes
    'okina_digraph',      # contains oʻ or gʻ
    'hamza_okina',        # standalone okina between vowels
    'russian_loan',       # known Russian borrowings patterns
    'arabic_persian_loan',# kitob-class
    'proper_noun_likely', # capitalized-ish patterns (rare in lowercased corpus)
    'tricky_clusters',    # unusual consonant sequences
]}

RU_MARKERS = ['tsiya','atsiya','rovka','izm','ist','tor','ura','yurt','kred','biuro','pro','kon','res','traktat']
ARABIC_PERSIAN = ['kitob','ilm','qalb','ilmu','adabiyot','tarix','sanʼat','shaʼni','maʼrifat','daʼvat']

for w in words:
    if len(w) < 3:
        continue
    if '\u02bb' in w:
        # digraph vs standalone hamza
        stripped = w.replace('o\u02bb','').replace('g\u02bb','')
        if '\u02bb' in stripped:
            strata['hamza_okina'].append(w)
        else:
            strata['okina_digraph'].append(w)
    elif any(m in w for m in RU_MARKERS):
        strata['russian_loan'].append(w)
    elif any(w.startswith(p) or p in w for p in ARABIC_PERSIAN[:6]):
        strata['arabic_persian_loan'].append(w)
    elif freq[w] >= 500:
        strata['high_freq_native'].append(w)
    else:
        strata['tricky_clusters'].append(w)

# allocation: weight toward tricky categories
alloc = {
    'high_freq_native': 120,
    'okina_digraph': 100,
    'hamza_okina': 80,
    'russian_loan': 60,
    'arabic_persian_loan': 40,
    'tricky_clusters': 100,
}

selected = []
for k, n in alloc.items():
    pool = sorted(set(strata[k]), key=lambda x: -freq.get(x, 0))
    # mix: take some frequent, some mid, some rare for diversity
    top = pool[:n//2]
    rest = [w for w in pool[n//2:] if freq.get(w,0) < 200]
    mid = random.sample(rest, min(n - len(top), len(rest)))
    chosen = list(dict.fromkeys(top + mid))[:n]
    selected.extend([(w, k) for w in chosen])

random.shuffle(selected)
selected = selected[:500]

# Now compute uzg2p output for each
import sys
sys.path.insert(0, '/home/ubuntu/uzbek-tts/uzg2p')
from uzg2p import G2P

# need patched epitran from g2p_env; use that env's site-packages
import os
g2p_env_site = '/home/ubuntu/uzbek-tts/g2p_env/lib/python3.11/site-packages'
sys.path.insert(0, g2p_env_site)
import epitran
epi = epitran.Epitran('uzb-Latn')

rows = []
for i, (w, cat) in enumerate(selected, 1):
    ph = epi.transliterate(w)
    rows.append((i, w, ph, cat, '', ''))  # empty columns: native_reading, verdict

out = '/home/ubuntu/uzbek-tts/g2p_audit/validation_set_500.tsv'
with open(out, 'w', encoding='utf-8', newline='') as f:
    wcsv = csv.writer(f, delimiter='\t')
    wcsv.writerow(['id', 'word', 'uzg2p_ipa', 'category', 'native_check_notes', 'verdict'])
    for r in rows:
        wcsv.writerow(r)

print(f'wrote {len(rows)} words to {out}')
from collections import Counter
print(Counter(r[3] for r in rows))
