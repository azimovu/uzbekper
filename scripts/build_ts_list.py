import re

words = []
with open('/home/ubuntu/uzbek-tts/g2p_audit/vocab_full.txt') as f:
    for line in f:
        w, c = line.rstrip('\n').split('\t')
        words.append((w, int(c)))

ts_words = [(w, c) for w, c in words if 'ts' in w]
print(f'words containing ts: {len(ts_words)}')

VOWELS = set('aeiou\u0259\u00f6\u00fc')

def is_intervocalic_ts(w):
    for m in re.finditer(r'ts', w):
        i = m.start()
        if i > 0 and i + 2 < len(w):
            if w[i-1] in VOWELS and w[i+2] in VOWELS:
                return True
    return False

keep = [(w, c) for w, c in ts_words if is_intervocalic_ts(w)]
to_s = [(w, c) for w, c in ts_words if not is_intervocalic_ts(w)]

print(f'\nINTERVOCALIC (keep /t+s/): {len(keep)}')
for w, c in sorted(keep, key=lambda x: -x[1])[:15]:
    print(f'  {w} ({c})')

print(f'\nMAPS TO /S/: {len(to_s)}')
for w, c in sorted(to_s, key=lambda x: -x[1])[:15]:
    print(f'  {w} ({c})')

with open('/home/ubuntu/uzbek-tts/uzg2p/uzg2p/ts_to_s_exceptions.txt', 'w') as f:
    for w, c in sorted(to_s):
        f.write(w + '\n')
print(f'\nwrote {len(to_s)} ts->s words to ts_to_s_exceptions.txt')
