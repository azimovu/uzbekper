import csv, random
random.seed(43)

rows = []
with open('/home/ubuntu/uzbek-tts/g2p_audit/validation_set_500.tsv', encoding='utf-8') as f:
    r = csv.reader(f, delimiter='\t')
    header = next(r)
    rows = list(r)

have = set(r[1] for r in rows)
vocab = []
with open('/home/ubuntu/uzbek-tts/g2p_audit/vocab_full.txt') as f:
    for line in f:
        w, c = line.rstrip('\n').split('\t')
        vocab.append((w, int(c)))

import epitran
epi = epitran.Epitran('uzb-Latn')

added = 0
for w, c in vocab[:3000]:
    if added >= 60:
        break
    if w in have or '\u02bb' in w or len(w) < 3:
        continue
    ph = epi.transliterate(w)
    rows.append([len(rows)+1, w, ph, 'high_freq_native_extra', '', ''])
    have.add(w)
    added += 1

with open('/home/ubuntu/uzbek-tts/g2p_audit/validation_set_500.tsv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(header)
    for r in rows:
        w.writerow(r)
print(f'total now: {len(rows)}')
