# GPT experiment outputs

`gpt_5_6_luna_none/` is the active formal protocol: five-arm calibration, GPT-generated and
fail-closed memory conditioned on that calibration, GPT-C evaluation, then a validated merge. Run or
resume it only through `experiments/04_gpt_luna_5repeat.sh`.

`gpt_5_6_luna_minimal/` is a superseded, incomplete pilot (118/270 arm-replicates at interruption).
It used an earlier low-effort adapter mapping, Claude-era performance conditioning, and the former
fail-open verifier. Preserve it as an audit artifact, but never resume it, merge it, or quote it as a
nine-screen result.

Codex-backed rows are evaluation/API replicates, not reproducible model seeds. The formal manifests
record the exact local runtime/input hashes, model identifier, reasoning setting, memory hash, and
`provider_seed: null`.
