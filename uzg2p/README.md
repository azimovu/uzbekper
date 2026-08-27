# uzg2p

Uzbek grapheme-to-phoneme conversion. Wraps [Epitran](https://github.com/dmort27/epitran)'s
`uzb-Latn` map with Uzbek-specific fixes:

- **Input normalization** — all apostrophe variants (ʻ ʼ ‘ ’ ' `) canonicalized
- **Standalone-okina fix** — `yaʻni` → /jaʔni/ (stock Epitran leaves the okina raw)
- **Foreign-word flagging** — heuristics mark likely non-Uzbek tokens (wellness, skil…)
- **Optional stress marking** — final-syllable default with exceptions dict
- **Optional strict oʻ** — /ɵ/ instead of Epitran's default /o/

## Install

```bash
pip install -e .
```

Note: requires the one-line Epitran map patch (standalone okina → glottal stop).
The patch is in this repo at `patches/uzb-Latn.diff`. Apply it to your installed
Epitran, or use `uzg2p.patch_epitran()` programmatically.

## Usage

```python
from uzg2p import G2P
g2p = G2P()
g2p("kitob").phonemes     # 'kit̪ɒb'
g2p("yaʻni").phonemes     # 'jaʔni'
g2p("yoʻq").phonemes      # 'joq'
```

## Tests

```bash
pytest tests/
```
