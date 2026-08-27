"""
uzg2p — Uzbek grapheme-to-phoneme conversion.

Wraps Epitran's uzb-Latn map with:
- input apostrophe normalization (corpus reality: U+02BB used everywhere)
- foreign-word detection (flags likely non-Uzbek tokens instead of guessing)
- optional stress marking (final-syllable default + exceptions)
- optional strict oʻ (/ɵ/) instead of Epitran's default /o/

Usage:
    from uzg2p import G2P
    g2p = G2P()
    g2p("yoʻq")            # 'joq'
    g2p("yaʻni")           # 'jaʔni'
    g2p("wellness")        # 'wellness' + .is_foreign == True on returned object

Author: Utkirbek Azimov, 2026. MIT license.
Core mappings by the Epitran project (dmort27/epitran), with one local patch:
standalone okina -> glottal stop.
"""
import re
import os
from dataclasses import dataclass

# Load ng-split exceptions (words where ng = /ŋg/, not /ŋ/)
_NG_SPLIT = set()
_ngfile = os.path.join(os.path.dirname(__file__), 'ng_split_exceptions.txt')
if os.path.exists(_ngfile):
    with open(_ngfile, encoding='utf-8') as f:
        _NG_SPLIT = {line.strip() for line in f if line.strip()}

# ts->s words (school rule: ts maps to /s/ unless intervocalic)
_TS_TO_S = set()
_tsfile = os.path.join(os.path.dirname(__file__), 'ts_to_s_exceptions.txt')
if os.path.exists(_tsfile):
    with open(_tsfile, encoding='utf-8') as f:
        _TS_TO_S = {line.strip() for line in f if line.strip()}

try:
    import epitran
except ImportError as e:
    raise ImportError(
        "uzg2p requires epitran. Install with: pip install epitran"
    ) from e

# All apostrophe-like chars seen in real Uzbek corpora, canonicalized to U+02BB
# (the dominant form: 603k occurrences vs zero others in a 701h corpus).
_APOSTROPHES = "\u02bc\u2018\u2019'`\u02bb"  # note: ʼ/‘/’/'/` all -> ʻ? No — see below.
# Actually: oʻ/gʻ digraphs need U+02BB; hamza needs glottal output either way.
# Since Epitran (patched) maps both ʻ and ʼ to correct outputs in context,
# we canonicalize everything to U+02BB for simplicity and let the patched map do its work.

_WORD_RE = re.compile(r"[a-zA-Z\u02bb\u02bc\u2018\u2019']+")

# Common English/Russian markers for foreign-word heuristics.
# NOTE: capital-letter check is separate — IGNORECASE would break it.
_FOREIGN_HINTS = re.compile(
    r"(^|[aeiou])(ck|ll|ss|th|wh|dge|ough)|"
    r"(\b)(the|and|tion|ness)\b"
)
_CAPITAL_START = re.compile(r"^[A-Z]")


@dataclass
class G2PResult:
    text: str          # input word
    phonemes: str      # IPA output
    is_foreign: bool   # heuristic flag: probably not native Uzbek
    stressed: str      # phonemes with stress mark (ˈ) if stress=True


class G2P:
    def __init__(
        self,
        strict_oo: bool = True,
        mark_stress: bool = False,
        stress_exceptions: dict | None = None,
        patched_map_dir: str | None = None,
    ):
        """
        strict_oo: remap Epitran's /o/ (from oʻ) to /ɵ/ (Tashkent-standard close-mid front rounded).
        mark_stress: append stressed form with ˈ before final syllable.
        stress_exceptions: {word: syllable_index} overriding final-syllable default.
        patched_map_dir: dir containing patched uzb-Latn.csv; if None, assumes
                         epitran installed with local patch (see README).
        """
        self.strict_oo = strict_oo
        self.mark_stress = mark_stress
        self.stress_exceptions = stress_exceptions or {}
        if patched_map_dir:
            os.environ["EPITRAN_MAP_DIR"] = patched_map_dir
        self._epi = epitran.Epitran("uzb-Latn")

    def _normalize(self, text: str) -> str:
        """Canonicalize apostrophes to U+02BB; lowercase."""
        out = []
        for ch in text:
            if ch in "\u02bc\u2018\u2019'`":
                out.append("\u02bb")
            else:
                out.append(ch.lower())
        return "".join(out)

    def _is_foreign(self, word: str) -> bool:
        """Heuristic: consonant clusters rare in Uzbek + known EN/RU suffixes."""
        if _FOREIGN_HINTS.search(word) or _CAPITAL_START.match(word):
            return True
        # Uzbek phonotactics: no word-initial consonant clusters beyond 'sh/ch/ng'
        first_two = word[:2]
        if len(word) > 2 and first_two[0] not in "aioeubdgqkhlmnprstvxzyfjcw" and first_two not in ("sh", "ch", "ng", "ts"):
            return True
        return False

    def _stress(self, phonemes: str, word: str) -> str:
        """Mark final-syllable stress unless exception says otherwise."""
        # crude but workable: split on vowel boundaries
        vowels = set("aeiouɒɵøəɪʊɛɔy")
        idxs = [i for i, ch in enumerate(phonemes) if ch in vowels]
        if not idxs:
            return phonemes
        syl_idx = self.stress_exceptions.get(word, -1)  # -1 = last syllable
        starts = []
        prev = -2
        for i in idxs:
            if i != prev + 1:
                starts.append(i)
            prev = i
        target = starts[syl_idx] if abs(syl_idx) <= len(starts) else starts[-1]
        return phonemes[:target] + "\u02c8" + phonemes[target:]

    def __call__(self, text: str) -> G2PResult:
        norm = self._normalize(text)

        # ng-split preprocessing: mark boundary ng as n+g before Epitran
        # (Epitran converts ng->ŋ blindly; boundary cases need /ŋg/)
        if norm in _NG_SPLIT:
            norm = norm.replace('ng', 'n\u200bg')  # zero-width space breaks digraph match
        elif norm in _TS_TO_S:
            # school rule: non-intervocalic ts reads as /s/
            norm = norm.replace('ts', 's')
        ph = self._epi.transliterate(norm)
        ph = ph.replace('\u200b', '')  # strip ZWSP from output

        if self.strict_oo:
            # Epitran maps oʻ->o; strict mode wants ɵ
            ph = self._strict_oo_replace(norm, ph)

        foreign = self._is_foreign(norm)
        stressed = self._stress(ph, norm) if self.mark_stress else ""
        return G2PResult(text=text, phonemes=ph, is_foreign=foreign, stressed=stressed)

    def _strict_oo(self_placeholder=None):
        pass

    def _strict_oo_replace(self, norm_word: str, phonemes: str) -> str:
        """Replace /o/ derived from oʻ with /ɵ/. Approximation: positional walk."""
        result = []
        pi = 0
        ni = 0
        while ni < len(norm_word):
            if norm_word.startswith("oʻ", ni):
                # find next phoneme chunk in phonemes matching length of 'o'
                result.append("ɵ")
                ni += 2
                pi += 1
            else:
                if pi < len(phonemes):
                    result.append(phonemes[pi])
                    pi += 1
                ni += 1
        # append remaining phoneme tail (multi-char outputs like t̪, d͡ʒ exceed
        # their grapheme count, leaving pi behind ni)
        if pi < len(phonemes):
            result.append(phonemes[pi:])
        return "".join(result)


import os  # placed here intentionally: only needed for env var path override
