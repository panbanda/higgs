#!/usr/bin/env python3
"""
Audio benchmark: vllm-mlx TTS and STT quality/performance.

mlx-server does not implement audio endpoints; this benchmarks vllm-mlx alone
and documents the feature gap.

Tests:
  1. TTS via /v1/audio/speech (Kokoro-82M) -- RTF and audio duration
  2. STT round-trip: TTS -> /v1/audio/transcriptions (Whisper) -> WER

Requirements:
    pip install mlx-audio
"""

import json
import signal
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

VLLM_PORT = 9003
PYTHON = sys.executable
LLAMA_1B = "mlx-community/Llama-3.2-1B-Instruct-4bit"
# Kokoro is downloaded on-demand by mlx-audio
TTS_MODEL = "mlx-community/Kokoro-82M-4bit"
STT_MODEL = "mlx-community/whisper-large-v3-turbo"

TTS_TEXTS = [
    ("short", "Hello, world! This is a test of text to speech synthesis."),
    ("medium", "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs."),
    ("long", (
        "Artificial intelligence is transforming the way we interact with computers. "
        "Large language models can now generate coherent text, answer complex questions, "
        "and assist with a wide range of tasks from coding to creative writing."
    )),
]


def wait_ready(port: int, timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    url = f"http://localhost:{port}/v1/models"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def stop(proc: subprocess.Popen):
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def post_json(port: int, path: str, body: dict) -> bytes:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def post_form(port: int, path: str, audio_bytes: bytes, filename: str, model: str) -> dict:
    """Multipart form upload for STT."""
    boundary = "BenchAudioBoundary7x9z"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="model"\r\n\r\n{model}\r\n'
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + audio_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"http://localhost:{port}{path}",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def wav_duration(wav_bytes: bytes) -> float:
    if len(wav_bytes) < 44:
        return 0.0
    sample_rate = struct.unpack_from("<I", wav_bytes, 24)[0]
    num_channels = struct.unpack_from("<H", wav_bytes, 22)[0]
    bits = struct.unpack_from("<H", wav_bytes, 34)[0]
    data_size = struct.unpack_from("<I", wav_bytes, 40)[0]
    samples = data_size // (num_channels * (bits // 8))
    return samples / sample_rate if sample_rate > 0 else 0.0


def word_error_rate(ref: str, hyp: str) -> float:
    r = ref.lower().split()
    h = hyp.lower().split()
    m, n = len(r), len(h)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[:], i
        for j in range(1, n + 1):
            dp[j] = prev[j - 1] if r[i - 1] == h[j - 1] else 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n] / max(m, 1)


def main():
    results: dict = {"tts": [], "stt": []}

    print("\n=== Audio Benchmark: TTS + STT (vllm-mlx) ===")
    print("Note: mlx-server does not implement audio endpoints.\n")

    # Start vllm-mlx with a base LLM (audio endpoints lazy-load TTS/STT models)
    proc = subprocess.Popen(
        [PYTHON, "-m", "vllm_mlx.cli", "serve", LLAMA_1B, "--port", str(VLLM_PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not wait_ready(VLLM_PORT):
        print("vllm-mlx failed to start -- aborting audio benchmark")
        stop(proc)
        return

    print("Server ready.\n")

    # --- TTS ---
    print("-- TTS (Kokoro-82M) --")
    tts_audio: list[tuple[bytes, str]] = []  # (audio_bytes, original_text)
    for test_id, text in TTS_TEXTS:
        print(f"  [{test_id}] len={len(text)} chars ...", end=" ", flush=True)
        try:
            t0 = time.perf_counter()
            audio = post_json(VLLM_PORT, "/v1/audio/speech", {
                "model": TTS_MODEL,
                "input": text,
                "voice": "af_heart",
                "response_format": "wav",
            })
            elapsed = time.perf_counter() - t0
            duration = wav_duration(audio)
            rtf = duration / elapsed if elapsed > 0 else 0
            print(f"{len(audio)//1024}KB, audio={duration:.1f}s, gen={elapsed:.2f}s, RTF={rtf:.1f}x")
            results["tts"].append({
                "id": test_id,
                "chars": len(text),
                "audio_kb": round(len(audio) / 1024, 1),
                "duration_s": round(duration, 2),
                "gen_s": round(elapsed, 2),
                "rtf": round(rtf, 1),
            })
            tts_audio.append((audio, text))
        except Exception as e:
            print(f"ERROR: {e}")
            results["tts"].append({"id": test_id, "error": str(e)})

    # --- STT round-trip ---
    print("\n-- STT round-trip (TTS audio -> Whisper transcription) --")
    for audio_bytes, reference in tts_audio:
        short_ref = reference[:50]
        print(f"  [{short_ref}...] ...", end=" ", flush=True)
        try:
            duration = wav_duration(audio_bytes)
            t0 = time.perf_counter()
            resp = post_form(VLLM_PORT, "/v1/audio/transcriptions",
                             audio_bytes, "audio.wav", STT_MODEL)
            elapsed = time.perf_counter() - t0
            transcript = resp.get("text", "").strip()
            rtf = duration / elapsed if elapsed > 0 else 0
            wer = word_error_rate(reference, transcript)
            print(f"WER={wer:.2f}, RTF={rtf:.1f}x")
            results["stt"].append({
                "reference": reference,
                "transcript": transcript,
                "wer": round(wer, 3),
                "rtf": round(rtf, 1),
                "elapsed_s": round(elapsed, 2),
            })
        except Exception as e:
            print(f"ERROR: {e}")
            results["stt"].append({"reference": reference[:60], "error": str(e)})

    stop(proc)

    # Print report
    print("\n\n=== AUDIO RESULTS ===\n")

    print("**TTS (Kokoro-82M, vllm-mlx only)**\n")
    print("| Test | Chars | Audio (s) | Gen time (s) | RTF |")
    print("|---|---|---|---|---|")
    for r in results["tts"]:
        if "error" not in r:
            print(f"| {r['id']} | {r['chars']} | {r['duration_s']} | {r['gen_s']} | {r['rtf']}x |")
        else:
            print(f"| {r['id']} | - | - | ERROR | - |")

    print("\n**STT round-trip (TTS -> Whisper large-v3-turbo)**\n")
    print("| Reference (truncated) | WER | RTF |")
    print("|---|---|---|")
    for r in results["stt"]:
        if "error" not in r:
            print(f"| {r['reference'][:50]} | {r['wer']} | {r['rtf']}x |")
        else:
            print(f"| {r.get('reference', '')[:50]} | ERROR | - |")

    print("\nmlx-server: audio endpoints not implemented (feature gap vs vllm-mlx).")

    # Save
    out_json = Path(__file__).parent / "audio_results.json"
    out_json.write_text(json.dumps(results, indent=2))

    out_md = Path(__file__).parent / "audio_results.md"
    lines = [
        "### Audio (vllm-mlx only)\n",
        "mlx-server does not implement audio endpoints. The following results are for vllm-mlx.\n",
        "**TTS: Kokoro-82M**\n",
        "| Input | Audio duration | Generation time | Real-time factor |",
        "|---|---|---|---|",
    ]
    for r in results["tts"]:
        if "error" not in r:
            lines.append(f"| {r['id']} ({r['chars']} chars) | {r['duration_s']}s | {r['gen_s']}s | {r['rtf']}x |")
        else:
            lines.append(f"| {r['id']} | ERROR | - | - |")

    lines += [
        "",
        "**STT round-trip: TTS audio -> Whisper large-v3-turbo transcription**\n",
        "| Input | Word error rate | Real-time factor |",
        "|---|---|---|",
    ]
    for r in results["stt"]:
        if "error" not in r:
            lines.append(f"| {r['reference'][:50]}... | {r['wer']} | {r['rtf']}x |")
        else:
            lines.append(f"| {r.get('reference', '')[:40]} | ERROR | - |")

    out_md.write_text("\n".join(lines))
    print(f"\nResults: {out_json}, {out_md}")


if __name__ == "__main__":
    main()
