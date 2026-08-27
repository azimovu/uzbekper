"""Tests for uzg2p — hand-verified against Sjoberg (1963) phoneme tables."""
import pytest
from uzg2p import G2P


@pytest.fixture(scope="module")
def g2p():
    return G2P()


class TestCore:
    def test_basic_word(self, g2p):
        # strict_oo default: final 'o' in kitob is plain o -> /ɒ/
        assert g2p("kitob").phonemes == "kit̪ɒb"

    def test_digraph_sh(self, g2p):
        assert g2p("shahar").phonemes.startswith("ʃa")

    def test_digraph_ch(self, g2p):
        assert "t͡ʃ" in g2p("choy").phonemes

    def test_digraph_ng(self, g2p):
        assert "ŋ" in g2p("ong").phonemes

    def test_gh_uvular(self, g2p):
        assert g2p("gʻazab").phonemes.startswith("ʁ")

    def test_glottal_okina(self, g2p):
        """yaʻni: standalone okina = glottal stop (patched map)."""
        assert "ʔ" in g2p("yaʻni").phonemes

    def test_apostrophe_canonicalization(self, g2p):
        """All apostrophe variants give same output."""
        variants = ["ya\u02bbni", "ya\u02bcni", "ya'ni", "ya\u2019ni"]
        outputs = {g2p(v).phonemes for v in variants}
        assert len(outputs) == 1


class TestStrictOo:
    def test_strict_mode_changes_o_gh_words(self):
        relaxed = G2P(strict_oo=False)
        strict = G2P(strict_oo=True)
        r = relaxed("yo\u02bbq").phonemes
        s = strict("yo\u02bbq").phonemes
        assert r != s          # they differ
        assert "\u0275" in s   # strict has close-mid front rounded


class TestForeign:
    def test_wellness_flagged(self, g2p):
        assert g2p("wellness").is_foreign is True

    def test_native_not_flagged(self, g2p):
        assert g2p("kitob").is_foreign is False


class TestStress:
    def test_final_syllable_default(self, g2p):
        g = G2P(mark_stress=True)
        result = g("kitob")
        assert "\u02c8" in result.stressed
        # stress on last syllable: before 'ɒ' (final 'b' dropped by map — known Epitran quirk)
        assert result.stressed.endswith("ˈɒb")

    def test_exception_override(self, g2p):
        g = G2P(mark_stress=True, stress_exceptions={"kitob": 0})
        result = g("kitob")
        # stress lands before the FIRST vowel: k + ˈi...
        assert result.stressed.startswith("k\u02c8i")
