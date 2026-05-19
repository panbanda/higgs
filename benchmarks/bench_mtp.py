#!/usr/bin/env python3
"""Sweep Higgs MTP draft depth for a local MLX model.

The benchmark compares baseline greedy decode with MTP disabled against
`HIGGS_MTP_DRAFT_N_MAX` values 1, 2, and 3. It starts a fresh Higgs server for
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
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", help="Local path or HF ID passed to `higgs serve --model`")
    parser.add_argument("--model-name", default=None, help="Request model name; defaults to path basename")
    parser.add_argument("--higgs-bin", default=os.environ.get("HIGGS_BIN", "./target/release/higgs"))
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=int, default=600)
    args = parser.parse_args()
    if args.model_name is None:
        args.model_name = os.path.basename(args.model_path.rstrip("/")) or args.model_path
    return args


def main() -> int:
    args = parse_args()
    trials = [
        Trial("baseline_mtp_off", {"HIGGS_MTP": "0"}),
        Trial("mtp_draft_1", {"HIGGS_MTP": "1", "HIGGS_MTP_DRAFT_N_MAX": "1"}),
        Trial("mtp_draft_2", {"HIGGS_MTP": "1", "HIGGS_MTP_DRAFT_N_MAX": "2"}),
        Trial("mtp_draft_3", {"HIGGS_MTP": "1", "HIGGS_MTP_DRAFT_N_MAX": "3"}),
    ]

    results = []
    for trial in trials:
        print(f"running {trial.label}...", flush=True)
        results.append(run_trial(args, trial))

    print(json.dumps({"model": args.model_path, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
