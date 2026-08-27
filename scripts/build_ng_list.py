import re, os

words = []
with open('/home/ubuntu/uzbek-tts/g2p_audit/vocab_full.txt') as f:
    for line in f:
        w, c = line.rstrip('\n').split('\t')
        words.append((w, int(c)))

SUFFIXES = ['lar', 'la', 'iga', 'ida', 'idan', 'i', 'ini', 'lari',
            'ligi', 'chan', 'gan', 'gani', 'siz', 'li', 'lik', 'dor', 'dagi']

split_words = set()
for w, c in words:
    if 'ng' not in w:
        continue
    # genitive -ning/-ing/-ung/-ong endings keep single ng (native-confirmed)
    if re.search(r'(ning|ing|ung|ong)$', w):
        continue
    idx = w.find('ng')
    rest = w[idx+2:]
    hit = any(rest.startswith(s) or rest == s for s in SUFFIXES)
    if re.search(r'ng(i|a|u)(r|s|m)?$', w):
        hit = True
    if hit:
        split_words.add(w)

outdir = '/home/ubuntu/uzbek-tts/uzg2p/uzg2p'
os.makedirs(outdir, exist_ok=True)
with open(f'{outdir}/ng_split_exceptions.txt', 'w') as f:
    for w in sorted(split_words):
        f.write(w + '\n')
print(f'wrote {len(split_words)} split-exception words to ng_split_exceptions.txt')
