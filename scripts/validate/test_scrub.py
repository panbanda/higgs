#!/usr/bin/env python3
"""Unit tests for validation report PII scrubbing."""

import unittest

from scrub import scrub


class ScrubTests(unittest.TestCase):
    def test_redacts_paths_users_hosts_network_and_secrets(self):
        source = """
path=/Users/alice/project /home/bob/work
user alice root
host buildbox.local buildbox buildbox.localdomain
network 192.168.1.25 2001:db8::1 aa:bb:cc:dd:ee:ff
keys sk-abcdefghijklmnopqrstuvwxyz123456 ghp_abcdefghijklmnopqrstuvwxyz123456 hf_abcdefghijklmnopqrstuvwxyz123456
cloud AKIAABCDEFGHIJKLMNOP xoxb-abcdefghijklmnop AGE-SECRET-KEY-1ABCDEFG
secret_token=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef1234567890
password: 0123456789abcdef0123456789abcdef
-----BEGIN PRIVATE KEY-----
very private
-----END PRIVATE KEY-----
email alice@example.com
"""
        cleaned = scrub(source, usernames={"alice", "root"}, hostnames={"buildbox", "buildbox.local", "buildbox.localdomain"})
        for value in (
            "/Users/alice", "/home/bob", "alice@example.com", "192.168.1.25",
            "2001:db8::1", "aa:bb:cc:dd:ee:ff", "sk-", "ghp_", "hf_", "AKIA",
            "xoxb-", "AGE-SECRET-KEY", "PRIVATE KEY", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef1234567890",
            "0123456789abcdef0123456789abcdef",
        ):
            self.assertNotIn(value, cleaned)
        self.assertIn("~", cleaned)
        self.assertIn("[user]", cleaned)
        self.assertIn("[host]", cleaned)
        self.assertIn("[email]", cleaned)

    def test_preserves_allowed_validation_data(self):
        source = """chip Apple M4 Max; RAM 64 GB; macOS 15.4
commit 0123456789abcdef0123456789abcdef01234567
model mlx-community/Qwen3-1.7B-4bit quant 4bit
decode_tokps 123.45 median_delta_pct -2.50
"""
        self.assertEqual(scrub(source, usernames=set(), hostnames=set()), source)


if __name__ == "__main__":
    unittest.main()
