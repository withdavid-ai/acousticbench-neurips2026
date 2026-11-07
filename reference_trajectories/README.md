# Sanitized reference scores

270 run files contain 198,300 rows for the main model roster, cascade and original
Kimi adapters. Each file retains the original question IDs, normalized answer
choices, gold keys, correctness, response status and task/split classification.
These are sufficient for the existing offline table generator.

Free-form responses are replaced by `[response withheld]` when originally
non-empty, or `""` when empty. This preserves parseability accounting without
publishing potentially identifying utterances or generated text. Error messages,
request payloads, question-set names and other nonessential metadata are removed.
**Do not run rejudging on these placeholders.** Rejudging requires newly collected
raw responses stored privately in `results/`.

Public question IDs correspond to the gated dataset. No audio, transcripts,
contributor/rater identities, private workspace paths or credentials are included.
Summary files are deliberately omitted; compute them from the row-level records.

```bash
python scripts/make_paper_tables.py --check
```

Historical option-order, cascade and extraction exceptions are documented in
[reproduction scope](../docs/reproduction.md). This is not a fresh inference run.
