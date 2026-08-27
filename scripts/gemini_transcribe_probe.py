"""Gemini 3.5 Transcribe batch probe — runs ON Modal (US egress) to bypass the
Warsaw geo gate. Downloads N real Uzbek Common Voice clips inside the
container, uploads each to the Gemini Files API, transcribes with a uz-UZ
hint, and returns all (clip, transcript) pairs for eyeballing viability.

Usage: modal run scripts/gemini_transcribe_probe.py --n 15
"""

import modal

app = modal.App("gemini-transcribe-probe")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("google-genai", "huggingface_hub")
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("google-gemini-staging")],
    timeout=900,
)
def probe(n: int = 15) -> list[dict]:
    import os
    import tarfile

    from google import genai
    from huggingface_hub import hf_hub_download

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    tar_path = hf_hub_download(
        repo_id="fsicoli/common_voice_17_0",
        filename="audio/uz/dev/uz_dev_0.tar",
        repo_type="dataset",
    )

    clips = []
    with tarfile.open(tar_path) as t:
        for m in t.getmembers():
            if m.isfile() and m.name.endswith((".mp3", ".wav")):
                clips.append((m.name, t.extractfile(m).read()))
            if len(clips) >= n:
                break

    results = []
    for idx, (name, data) in enumerate(clips):
        suf = ".mp3" if name.endswith(".mp3") else ".wav"
        mime = "audio/mpeg" if suf == ".mp3" else "audio/wav"
        import tempfile
        import time

        tmp = tempfile.NamedTemporaryFile(suffix=suf, delete=False)
        tmp.write(data)
        tmp.flush()

        up = client.files.upload(file=tmp.name, config={"mime_type": mime})
        inter = client.interactions.create(
            model="gemini-3.5-transcribe",
            input=[{"type": "audio", "uri": up.uri, "mime_type": mime}],
            generation_config={
                "transcription_config": {"language_codes": ["uz-UZ"]}
            },
        )
        results.append({"clip": name, "text": inter.output_text})
        # Free the uploaded blob so we don't accrue Files-API storage.
        try:
            client.files.delete(name=up.name)
        except Exception:
            pass
        # Free-tier limit: 10 req/min/model. Sleep to stay under it.
        if idx < len(clips) - 1:
            time.sleep(7)
    return results


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("google-gemini-staging")],
    timeout=300,
)
def delete_all_files() -> int:
    import os

    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    deleted = 0
    for f in client.files.list():
        client.files.delete(name=f.name)
        deleted += 1
    return deleted


@app.local_entrypoint()
def main(n: int = 15):
    for r in probe.remote(n):
        print(r)


@app.local_entrypoint()
def cleanup():
    """Delete all uploaded Gemini Files-API blobs (free storage)."""
    print(f"deleted {delete_all_files.remote()} files")
