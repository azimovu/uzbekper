"""WER computation for the Gemini 3.5 Transcribe 15-clip probe.

Tokenizes (lowercase, strip punctuation, whitespace split — Uzbek-aware:
the CV apostrophe in o'/g'/k' is treated as a normal char boundary, not
removed, matching Common Voice reference tokenization). Edit-distance based
word error rate: WER = (S + D + I) / N over the reference.
"""
import json
import re
import sys

# gemini transcripts from the Modal batch run (clip_id -> output_text)
GEMINI = {
    "common_voice_uz_27123210.mp3": "Yana kutib o'tirish niyatim yo'q.",
    "common_voice_uz_27123293.mp3": "Futbolimiz tarixida ham uning o'z o'rni bor va ancha yuqorida.",
    "common_voice_uz_27123295.mp3": "Birinchi marta sovg'a keltirib berganingizda onangizning ko'zida qalqigan yoshmi?",
    "common_voice_uz_27123337.mp3": "Amerikada haydovchilik guvohnomasini olish juda oson.",
    "common_voice_uz_27123369.mp3": "Kelajakda ham shunday bo'lib qolishini xohlar edim.",
    "common_voice_uz_27123370.mp3": "Unga doimo va abadiy hamd bo'lsin.",
    "common_voice_uz_27123418.mp3": "Insonlarga muvaffaqiyat tarqatish yaxshiroqmi yoki ulug'vor qadriyatlar?",
    "common_voice_uz_27123420.mp3": "Bunga javoban Isroil ham G'azoni bombalashga tushib ketadi.",
    "common_voice_uz_27123421.mp3": "Bunday fikrlashning ildizi juda insoniyatga xos.",
    "common_voice_uz_27123422.mp3": "Lekin ularning vazifasi va mazmuniga e'tibor bermaslik oqibatida xatolarga yo'l qo'yamiz.",
    "common_voice_uz_27123453.mp3": "O'zbeklar butun janubiy hududlarning ijtimoiy va madaniy hayotida faol ishtirok etardi.",
    "common_voice_uz_27123455.mp3": "Provokatsiyalarga uchmang, fitnalardan saqlaning.",
    "common_voice_uz_27123456.mp3": "Bir og'iz shirin soz, bir chimdim mehr kimni o'ldiribdi?",
    "common_voice_uz_27123539.mp3": "Qanday qilib?",
    "common_voice_uz_27123545.mp3": "Tayyor bo'lganingizdan keyin imtihon topshirib, o'z guvohnomangizni olib ketasiz.",
}

PUNCT = re.compile(r"[^\w\s']", re.UNICODE)


def tok(s: str) -> list[str]:
    s = s.lower()
    s = PUNCT.sub(" ", s)
    return [w for w in s.split() if w]


def wer(ref: list[str], hyp: list[str]) -> tuple[int, int, int, int, float]:
    # Levenshtein on word lists.
    n, m = len(ref), len(hyp)
    d = list(range(m + 1))
    for i in range(1, n + 1):
        prev = d[0]
        d[0] = i
        for j in range(1, m + 1):
            cur = d[j]
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + cost)
            prev = cur
    dist = d[m]
    S = D = I = 0
    # backtrack for counts
    i, j = n, m
    dd = [[0] * (m + 1) for _ in range(n + 1)]
    for a in range(n + 1):
        dd[a][0] = a
    for b in range(m + 1):
        dd[0][b] = b
    for a in range(1, n + 1):
        for b in range(1, m + 1):
            c = 0 if ref[a - 1] == hyp[b - 1] else 1
            dd[a][b] = min(dd[a - 1][b] + 1, dd[a][b - 1] + 1, dd[a - 1][b - 1] + c)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dd[i][j] == dd[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dd[i][j] == dd[i - 1][j - 1] + 1:
            S += 1; i, j = i - 1, j - 1
        elif i > 0 and dd[i][j] == dd[i - 1][j] + 1:
            D += 1; i -= 1
        else:
            I += 1; j -= 1
    n_ref = len(ref)
    return S, D, I, n_ref, (S + D + I) / n_ref if n_ref else 0.0


def main() -> int:
    refs = json.load(open("/tmp/cv_refs.json", encoding="utf-8"))
    total_s = total_d = total_i = total_n = 0
    rows = []
    for cid, hyp in GEMINI.items():
        ref = refs.get(cid)
        if ref is None:
            continue
        s, d, i, n, w = wer(tok(ref), tok(hyp))
        total_s += s; total_d += d; total_i += i; total_n += n
        rows.append((cid, s, d, i, n, w))
    print(f"{'clip':<32}{'S':>3}{'D':>3}{'I':>3}{'N':>4}{'WER':>8}")
    for cid, s, d, i, n, w in rows:
        print(f"{cid:<32}{s:>3}{d:>3}{i:>3}{n:>4}{w*100:>7.1f}%")
    agg = (total_s + total_d + total_i) / total_n if total_n else 0
    print("-" * 54)
    print(f"{'AGGREGATE (15 clips)':<32}{total_s:>3}{total_d:>3}{total_i:>3}{total_n:>4}{agg*100:>7.1f}%")
    print(f"\nTotal ref words: {total_n} | Sub={total_s} Del={total_d} Ins={total_i}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
