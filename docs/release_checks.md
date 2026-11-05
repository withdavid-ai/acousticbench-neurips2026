# Release checks — 2026-11-04

This code release was assembled in a fresh clone, without importing private
research repository history. Original sources were retained separately.

## Published content

- Evaluation harness, scoring and model serving implementations.
- Available Kimi/Qwen research tooling, with scope described in reproduction.md.
- 270 sanitized reference files containing 198,300 per-item score records.
- Dataset Croissant metadata; gated audio and transcripts are not in this repository.
- Tests, dependency specifications, licenses and setup documentation.

Reference records use an explicit field allowlist. Raw responses, error payloads,
prompts and internal metadata were removed. Response-presence placeholders cannot
be used to rerun answer extraction. Public author credits, upstream attribution
and the public research contact remain intentionally.

## Verification

- 65 offline tests passed.
- Python compilation and Ruff undefined-name/syntax checks passed.
- Table regeneration: 550 published cells matched; zero missing or mismatched cells.
- CLI help smoke tests passed for the research runners and model/task listings.
- Kimi training launcher and Kimi/Qwen checkpoint serving modules imported locally.
- Staged dataset validation: 5,835 checksum entries verified; 5,800 audio slots,
  5,355 unique recordings, 11.607 hours summed across slots; no hard failures.
- Dataset warnings are documented historical reference-ID/key differences and
  conservative metadata heuristics (public contact/URLs, publisher names, A/B labels).
- Publishable-file scan found no credential/private-path violations or raw responses.
- An additional detect-secrets scan was reviewed: 54 checksum/revision findings and
  four named-secret identifiers; no credential values found.
- Email review found only the public research contact and a synthetic test fixture.

No new paid inference, GPU training, or remote deployment was performed. These
checks do not establish the provenance of unavailable experimental records, or
guarantee that every upstream dependency will run on every GPU environment.
