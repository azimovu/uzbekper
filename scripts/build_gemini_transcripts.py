"""Build transcripts_gemini.json for the uzbekper scorer from the 15-clip
Gemini 3.5 Transcribe probe, then compute WER + PER via the project's own
benchmark_score.py (uzg2p G2P, comparable to Whisper/NeMo numbers).

Input: /tmp/cv_refs.json (CV validated.tsv references) + hardcoded GEMINI map
(from scripts/gemini_wer.py batch output, 2026-08-27).
Output: final/transcripts_gemini.json
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

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

refs = json.load(open("/tmp/cv_refs.json", encoding="utf-8"))

rows = []
missing = []
for cid, hyp in GEMINI.items():
    ref = refs.get(cid)
    if ref is None:
        missing.append(cid)
        continue
    rows.append({"ref": ref, "hyp": hyp, "source": "common_voice"})

if missing:
    print(f"WARNING: {len(missing)} clips missing refs: {missing}")

out_path = os.path.join(HERE, "..", "final", "transcripts_gemini.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
json.dump(rows, open(out_path, "w"), ensure_ascii=False, indent=2)
print(f"wrote {len(rows)} rows -> {out_path}")
