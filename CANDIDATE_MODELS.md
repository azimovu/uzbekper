# Candidate ASR models for UzbekPER benchmark (found 2026-08-25 via search)

NOT yet vetted — treat as leads only. Vet like GitNazarov before adding to any run
(check: actual base model in config.json vs card claim, <|uz|> token presence,
language-pin behavior, training data provenance).

| Model | Base (claimed) | Notes |
|---|---|---|
| Gearnode/qwen3-asr-uzbek | Qwen/Qwen3-ASR-1.7B | New arch family for the bench; would need its own inference path |
| Gearnode/qwen3-asr-uzbek-v2 | Qwen/Qwen3-ASR-1.7B | v2, same base |
| Kotib/uzbek_stt_v1 | Whisper Medium | Kotibai & Rubai Team |
| aisha-org/Whisper-Uzbek | openai/whisper-medium | Trained on Common Voice 17.0 |
| mirodil/whisper-small-uzbek | openai/whisper-small | CV-based, same family as BlueRaccoon baseline |

Existing roster (from SESSION_HANDOFF.md):
- BlueRaccoon/whisper-small-uz — only clean result so far (WER ~30% on 4.2k)
- openai/whisper-large-v3 — has <|uz|> token; re-run with verified language pin
- GitNazarov/whisper-large-uz — DROPPED (no <|uz|> token, mislabeled card)
- nvidia/stt_uz_fastconformer_hybrid_large_pc — never ran (NeMo env clash)
- uzinfocom-edu-ai/asr-uz-fastconformer-large — never ran

Vet checklist (per model):
1. config.json `_name_or_path` matches claimed base
2. tokenizer has Uzbek token (whisper) / no Turkish-only output on 5 test clips
3. model card discloses training data + license
4. weight file size consistent with claimed architecture
