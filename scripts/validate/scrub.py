#!/usr/bin/env python3
"""Remove personally identifying and credential-like data from validation output."""

import getpass
import os
import re
import socket
import subprocess
import sys


def local_hostnames():
    names = {socket.gethostname(), socket.getfqdn()}
    try:
        value = subprocess.run(
            ["scutil", "--get", "LocalHostName"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if value:
            names.add(value)
    except FileNotFoundError:
        pass
    return {name for name in names if name}


def scrub(text, usernames=None, hostnames=None):
    """Return text with PII and secrets replaced by stable, non-sensitive markers."""
    usernames = set(usernames) if usernames is not None else {getpass.getuser(), os.environ.get("USER", "")}
    hostnames = set(hostnames) if hostnames is not None else local_hostnames()

    text = re.sub(r"-----BEGIN [^-\n]+-----.*?-----END [^-\n]+-----", "[pem]", text, flags=re.DOTALL)
    text = re.sub(r"/(?:Users|home)/[^/\s]+", "~", text)
    text = re.sub(r"\b(?:sk-[A-Za-z0-9_-]+|ghp_[A-Za-z0-9]+|hf_[A-Za-z0-9_-]+|AKIA[A-Z0-9]{16}|xox[bpars]-[A-Za-z0-9-]+|AGE-SECRET-KEY-[A-Za-z0-9-]+)\b", "[secret]", text)
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[ip]", text)
    text = re.sub(r"(?<![\w:])(?:[0-9A-Fa-f]{1,4}:){2,}[0-9A-Fa-f:]+(?![\w:])", "[ip]", text)
    text = re.sub(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b", "[mac]", text)
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[email]", text)
    text = re.sub(
        r"(?i)\b((?:key|token|secret|password)(?:_[A-Za-z0-9-]+)?)([\s:=]+)([A-Za-z0-9+/=_-]{32,})",
        lambda match: f"{match.group(1)}{match.group(2)}[secret]",
        text,
    )
    for username in sorted(filter(None, usernames), key=len, reverse=True):
        text = re.sub(rf"(?<![A-Za-z0-9_.-]){re.escape(username)}(?![A-Za-z0-9_.-])", "[user]", text, flags=re.IGNORECASE)
    for hostname in sorted(filter(None, hostnames), key=len, reverse=True):
        text = re.sub(rf"(?<![A-Za-z0-9_.-]){re.escape(hostname)}(?![A-Za-z0-9_.-])", "[host]", text, flags=re.IGNORECASE)
    return text


def main():
    sys.stdout.write(scrub(sys.stdin.read()))


if __name__ == "__main__":
    main()
