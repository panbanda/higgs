#!/usr/bin/env python3
"""Sweep Higgs speculative decode modes for a local MLX model.

The benchmark compares baseline greedy decode against built-in MTP draft depth
and prompt-lookup speculative decode trials. It starts a fresh Higgs server for
each run so model/runtime env is isolated.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = (
    "Write a concise technical explanation of speculative decoding for local "
    "LLM inference. Include acceptance rate, verification cost, and why greedy "
    "decode is the easiest correctness target."
)


@dataclass
class Trial:
    label: str
    env: dict[str, str]


def derive_model_name(model_path: str) -> str:
    path = Path(model_path.rstrip("/"))
    parts = path.parts
    if "snapshots" in parts:
        idx = parts.index("snapshots")
        if idx > 0:
            cache_name = parts[idx - 1]
            if cache_name.startswith("models--"):
                name = cache_name.removeprefix("models--").replace("--", "/")
                if name:
                    return name
    return path.name or model_path


def wait_for_server(base: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(0.25)
    raise RuntimeError(f"server did not become ready: {last_error}")


def stop_server(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=15)


def api_request(base: str, body: dict[str, Any], timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def run_trial(args: argparse.Namespace, trial: Trial) -> dict[str, Any]:
    base = f"http://127.0.0.1:{args.port}"
    env = os.environ.copy()
    env.update(trial.env)

    cmd = [
        args.higgs_bin,
        "serve",
        "--model",
        args.model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--mlx-profile",
        "throughput",
    ]
    show_server_logs = os.environ.get("HIGGS_BENCH_SHOW_SERVER_LOGS") == "1"
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE if show_server_logs else subprocess.DEVNULL,
        stderr=subprocess.PIPE if show_server_logs else subprocess.DEVNULL,
    )
    try:
        wait_for_server(base, args.startup_timeout)
        body = {
            "model": args.model_name,
            "messages": [{"role": "user", "content": args.prompt}],
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "stream": False,
        }
        started = time.perf_counter()
        payload = api_request(base, body, args.request_timeout)
        elapsed = time.perf_counter() - started
    finally:
        stop_server(proc)

    server_log = ""
    if show_server_logs:
        chunks = []
        if proc.stdout is not None:
            chunks.append(proc.stdout.read().decode(errors="replace"))
        if proc.stderr is not None:
            chunks.append(proc.stderr.read().decode(errors="replace"))
        server_log = "\n".join(chunks)

    usage = payload.get("usage", {})
    completion_tokens = int(usage.get("completion_tokens", 0))
    tok_s = completion_tokens / elapsed if elapsed > 0 else 0.0
    content = (
        payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    return {
        "label": trial.label,
        "elapsed_s": elapsed,
        "completion_tokens": completion_tokens,
        "tok_s": tok_s,
        "content_prefix": content[:120],
        "server_log": "\n".join(
            line
            for line in server_log.splitlines()
            if "MTP decode complete" in line
            or "Prompt-lookup decode complete" in line
            or "Engine ready" in line
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", help="Local path or HF ID passed to `higgs serve --model`")
    parser.add_argument("--model-name", default=None, help="Request model name; defaults to path basename")
    parser.add_argument("--higgs-bin", default=os.environ.get("HIGGS_BIN", "./target/release/higgs"))
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--trials",
        default="baseline,1,2,3",
        help="Comma-separated trials: baseline and/or draft depths like 1,2,3",
    )
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=int, default=600)
    args = parser.parse_args()
    if args.model_name is None:
        args.model_name = derive_model_name(args.model_path)
    return args


def main() -> int:
    args = parse_args()
    trials = []
    for trial in args.trials.split(","):
        trial = trial.strip()
        if trial == "baseline":
            trials.append(Trial("baseline_mtp_off", {"HIGGS_MTP": "0"}))
        elif trial in {"mtp_default", "default"}:
            trials.append(Trial("mtp_default", {"HIGGS_MTP": "1"}))
        elif trial in {"prompt_lookup", "plookup"}:
            trials.append(
                Trial(
                    "prompt_lookup",
                    {"HIGGS_MTP": "0", "HIGGS_PROMPT_LOOKUP": "1"},
                )
            )
        elif trial in {"prompt_lookup_unchecked", "plookup_unchecked"}:
            trials.append(
                Trial(
                    "prompt_lookup_unchecked",
                    {
                        "HIGGS_MTP": "0",
                        "HIGGS_PROMPT_LOOKUP": "1",
                        "HIGGS_PROMPT_LOOKUP_UNCHECKED": "1",
                    },
                )
            )
        elif trial:
            depth = int(trial)
            trials.append(
                Trial(
                    f"mtp_draft_{depth}",
                    {"HIGGS_MTP": "1", "HIGGS_MTP_DRAFT_N_MAX": str(depth)},
                )
            )

    results = []
    for trial in trials:
        print(f"running {trial.label}...", flush=True)
        results.append(run_trial(args, trial))

    print(json.dumps({"model": args.model_path, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
