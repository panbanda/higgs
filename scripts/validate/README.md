# Validation scripts

Run a validation suite from the repository root:

```sh
scripts/validate/run.sh aa-smoke
scripts/validate/run.sh --baseline origin/main aa-smoke
```

The runner detects the Mac chip, RAM, and macOS version. It chooses a `small`
model below 32 GB RAM, a `medium` model from 32 through 64 GB, and the first
manifest model above 64 GB. Models are downloaded resumably into
the standard Hugging Face hub cache (respecting `HF_HOME` and
`HF_HUB_CACHE`). Baseline and candidate release binaries are cached under
`~/.cache/higgs-validate/builds/<sha>/`; their Cargo build output is shared at
`~/.cache/higgs-validate/target/` so unchanged dependencies can be reused.

`aa-smoke` starts each built server in turn and uses `bench_decode` with five
measured trials by default. Set `RUNS` to change that count and
`ACCEPTANCE_THRESHOLD` to change the allowed median regression percentage.

Each suite writes committed artifacts to `validation/<pr-id>/`:

- `report.md` contains summary statistics, the median delta, and the verdict.
- `runs.csv` contains each decode throughput measurement.

The ignored `validation/<pr-id>/raw/` directory is scratch input and logs.
Run `scripts/validate/run.sh --self-test` for an offline end-to-end check of
report rendering and PII scrubbing.

## PII policy

Reports may contain the chip model, RAM, macOS version, commit SHAs, model
names and quantizations, and benchmark metrics. They must never contain
usernames, home paths, hostnames, network information, environment dumps, or
keys. `scrub.py` filters report and CSV output before it is written; its unit
tests cover the redaction rules.
