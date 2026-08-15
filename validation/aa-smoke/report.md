# Validation report: aa-smoke

- chip: Apple M4 Max
- ram_gb: 128
- macos_version: 26.5.1
- baseline_sha: 853a054583fa7025c0ca01fe6ff4946d58a89b7a
- candidate_sha: b6bb89b8263e2598ee7987e34c235f5a248e9728
- model: qwen3-1.7B-4bit

| Side | Median decode tok/s | Min | Max | Stdev |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 319.62 | 317.78 | 320.31 | 0.96 |
| Candidate | 319.35 | 318.42 | 319.71 | 0.59 |

Verdict: **PASS** — median delta -0.09% (acceptance threshold -5.00%).
