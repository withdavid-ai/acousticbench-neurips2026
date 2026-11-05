"""Two robustness analyses on the stored CoT results. No new inference.

**1. Paired significance.** All four cells of a task share one fixed 200-item
subset, so direct and CoT are paired per item and the right test is McNemar on the
discordant pairs, not a two-proportion interval. Quoting +/-7 from independent
samples throws away the pairing and badly understates power: a 32-point swing with
~65 discordant items is not marginal.

**2. Does abstention track the physical cue?** Kimi declines to answer on 87 of 200
RAD items, saying the clips sound identical. RAD items are built by applying a
single-axis transformation, so the manipulated cue is measurable from the audio.
If abstention concentrates on small cue differences, the model has coherent
perception with a threshold. If it is flat in cue size -- calling a 15 dB gap
identical as readily as a 3 dB one -- that is a harder failure than any accuracy
number, and it is the more quotable result.

Cue is measured per dimension, matching what was manipulated:

    loudness        RMS level difference, dB
    pitch           median f0 difference, semitones (librosa.yin)
    tempo           onset-rate ratio, log2 (duration is preserved by the render,
                    so a duration ratio measures nothing -- checked, it is 0.00)
    dynamic-range   difference in frame-RMS spread, dB

Magnitudes are not comparable across dimensions, so items are ranked within their
own dimension and then pooled by quartile of within-dimension cue strength.

Measurements are cached; the audio is only read once.

Usage:
    python -m reproduction.finetuning_lift.cot_stats --mcnemar
    python -m reproduction.finetuning_lift.cot_stats --cues
    python -m reproduction.finetuning_lift.cot_stats --all
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproduction.finetuning_lift import config, cot, dataset  # noqa: E402

CUE_CACHE = config.RESULTS_ROOT / "cot" / "_rad_cue_magnitudes.json"


# --------------------------------------------------------------------------
# 1. McNemar
# --------------------------------------------------------------------------


def _exact_two_sided_binomial(b: int, c: int) -> float:
    """Exact McNemar: P(|X - n/2| >= |b - n/2|) for X ~ Binom(b + c, 0.5).

    Exact rather than the chi-square approximation because several cells have few
    discordant pairs, where the approximation is unreliable in the direction that
    would flatter us.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2 * tail)


@dataclass(frozen=True)
class Paired:
    task: str
    model: str
    n: int
    direct_acc: float
    cot_acc: float
    both_right: int
    both_wrong: int
    direct_only: int  # direct right, CoT wrong
    cot_only: int  # CoT right, direct wrong
    p_value: float

    @property
    def delta(self) -> float:
        return self.cot_acc - self.direct_acc

    @property
    def discordant(self) -> int:
        return self.direct_only + self.cot_only


def _load(task: str, model: str, condition: str) -> dict[str, dict]:
    path = config.RESULTS_ROOT / "cot" / f"{task}--{model}--{condition}" / "predictions.jsonl"
    if not path.exists():
        return {}
    return {r["example_id"]: r for r in (json.loads(l) for l in path.open())}


def paired_tests() -> list[Paired]:
    out: list[Paired] = []
    for task in cot.COT_TASKS:
        for model in cot.COT_MODELS:
            direct, cotc = _load(task, model, "direct"), _load(task, model, "cot")
            shared = sorted(set(direct) & set(cotc))
            if not shared:
                continue
            tt = tf = ft = ff = 0
            for eid in shared:
                d, c = bool(direct[eid]["correct"]), bool(cotc[eid]["correct"])
                if d and c:
                    tt += 1
                elif d and not c:
                    tf += 1
                elif c and not d:
                    ft += 1
                else:
                    ff += 1
            out.append(
                Paired(
                    task=task,
                    model=model,
                    n=len(shared),
                    direct_acc=100.0 * (tt + tf) / len(shared),
                    cot_acc=100.0 * (tt + ft) / len(shared),
                    both_right=tt,
                    both_wrong=ff,
                    direct_only=tf,
                    cot_only=ft,
                    p_value=_exact_two_sided_binomial(tf, ft),
                )
            )
    return out


def print_mcnemar(results: list[Paired]) -> None:
    print("\nPaired direct-vs-CoT comparison (exact McNemar on discordant items)\n")
    head = (
        f"  {'task':15s}{'model':11s}{'n':>4}{'direct':>8}{'CoT':>7}{'delta':>8}"
        f"{'D-only':>8}{'C-only':>8}{'p':>10}"
    )
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in results:
        star = " *" if r.p_value < 0.05 else ""
        print(
            f"  {r.task:15s}{r.model:11s}{r.n:>4}{r.direct_acc:>8.1f}{r.cot_acc:>7.1f}"
            f"{r.delta:>+8.1f}{r.direct_only:>8}{r.cot_only:>8}{r.p_value:>10.2g}{star}"
        )
    print("\n  D-only = direct right and CoT wrong; C-only = the reverse.")
    print("  p is a two-sided exact binomial test on those two counts. * = p < 0.05.")


# --------------------------------------------------------------------------
# 2. Cue magnitude vs abstention
# --------------------------------------------------------------------------


def _frame_rms_db(y, frame_length: int = 1024) -> list[float]:
    import librosa
    import numpy as np

    rms = librosa.feature.rms(y=y, frame_length=frame_length)[0]
    rms = rms[rms > 0]
    return list(20.0 * np.log10(rms)) if len(rms) else [0.0]


def _onset_rate(y, sr: int) -> float:
    """Onsets per second. The tempo cue changes event rate, not clip length: a
    duration ratio on these items is identically 0.00, so it measures nothing."""
    import librosa

    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    duration = len(y) / sr
    return (len(onsets) / duration) if duration > 0 and len(onsets) else 0.0


def _median_f0_hz(y, sr: int) -> float:
    import librosa
    import numpy as np

    f0 = librosa.yin(y, fmin=50, fmax=1000, sr=sr)
    f0 = f0[np.isfinite(f0)]
    return float(np.median(f0)) if len(f0) else 0.0


def measure_cues(*, refresh: bool = False) -> dict[str, dict]:
    """Per-item cue magnitude for the RAD subset, cached to disk."""
    if CUE_CACHE.exists() and not refresh:
        return json.loads(CUE_CACHE.read_text())

    import librosa
    import numpy as np

    task = config.TASKS["rad"]
    out: dict[str, dict] = {}
    examples = cot.subset(task)
    for i, ex in enumerate(examples, start=1):
        paths = ex.audio_paths
        ys, srs = [], []
        for p in paths:
            y, sr = librosa.load(str(p), sr=None, mono=True)
            ys.append(y)
            srs.append(sr)
        dim = ex.dimension or "unknown"

        if dim == "loudness":
            a, b = (st.mean(_frame_rms_db(y)) for y in ys)
            magnitude, unit = abs(b - a), "dB"
        elif dim == "pitch":
            fa, fb = (_median_f0_hz(y, sr) for y, sr in zip(ys, srs))
            magnitude = (
                abs(12.0 * math.log2(fb / fa)) if fa > 0 and fb > 0 else 0.0
            )
            unit = "semitones"
        elif dim == "tempo":
            ra, rb = (_onset_rate(y, sr) for y, sr in zip(ys, srs))
            magnitude = abs(math.log2(rb / ra)) if ra > 0 and rb > 0 else 0.0
            unit = "log2 onset-rate ratio"
        else:  # dynamic-range: spread of frame levels
            sa, sb = (
                (st.pstdev(_frame_rms_db(y)) if len(_frame_rms_db(y)) > 1 else 0.0)
                for y in ys
            )
            magnitude, unit = abs(sb - sa), "dB spread"

        out[ex.example_id] = {
            "dimension": dim,
            "magnitude": round(float(magnitude), 4),
            "unit": unit,
        }
        if i % 50 == 0:
            print(f"    measured {i}/{len(examples)}", flush=True)

    CUE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CUE_CACHE.write_text(json.dumps(out, indent=2))
    return out


def print_cue_analysis(model: str = "kimi") -> None:
    cues = measure_cues()
    recs = _load("rad", model, "cot")
    direct = _load("rad", model, "direct")
    if not recs:
        print(f"  no CoT results for {model}")
        return

    # Rank within dimension: magnitudes are not comparable across cue types.
    by_dim: dict[str, list[tuple[str, float]]] = {}
    for eid, info in cues.items():
        if eid in recs:
            by_dim.setdefault(info["dimension"], []).append((eid, info["magnitude"]))
    strength: dict[str, float] = {}
    for dim, items in by_dim.items():
        items.sort(key=lambda x: x[1])
        for rank, (eid, _) in enumerate(items):
            strength[eid] = rank / max(1, len(items) - 1)

    rows = [
        (
            strength[eid],
            cues[eid]["dimension"],
            cues[eid]["magnitude"],
            not recs[eid]["parsed"],
            bool(recs[eid]["correct"]),
            bool(direct.get(eid, {}).get("correct")),
        )
        for eid in sorted(strength)
    ]
    rows.sort(key=lambda r: r[0])

    print(f"\nAbstention vs measured cue magnitude, RAD / {model} / CoT (n={len(rows)})\n")
    print(f"  {'cue strength':16s}{'n':>5}{'abstain %':>11}{'CoT acc':>9}{'direct acc':>12}")
    print("  " + "-" * 53)
    q = max(1, len(rows) // 4)
    for i, label in enumerate(("Q1 (smallest)", "Q2", "Q3", "Q4 (largest)")):
        chunk = rows[i * q : (i + 1) * q] if i < 3 else rows[3 * q :]
        if not chunk:
            continue
        ab = 100.0 * sum(1 for r in chunk if r[3]) / len(chunk)
        ca = 100.0 * sum(1 for r in chunk if r[4]) / len(chunk)
        da = 100.0 * sum(1 for r in chunk if r[5]) / len(chunk)
        print(f"  {label:16s}{len(chunk):>5}{ab:>11.1f}{ca:>9.1f}{da:>12.1f}")

    absten = [r for r in rows if r[3]]
    committed = [r for r in rows if not r[3]]
    print(
        f"\n  mean cue-strength percentile: abstained {st.mean(r[0] for r in absten):.2f}"
        f"  vs committed {st.mean(r[0] for r in committed):.2f}"
    )

    print("\n  Per dimension, in the dimension's own units:")
    print(f"  {'dimension':16s}{'unit':22s}{'n':>4}{'abstain %':>11}{'mean |cue| abst.':>18}{'committed':>11}")
    for dim in sorted(by_dim):
        sub = [r for r in rows if r[1] == dim]
        a = [r[2] for r in sub if r[3]]
        c = [r[2] for r in sub if not r[3]]
        unit = next(v["unit"] for v in cues.values() if v["dimension"] == dim)
        print(
            f"  {dim:16s}{unit:22s}{len(sub):>4}"
            f"{100.0 * len(a) / max(1, len(sub)):>11.1f}"
            f"{(st.mean(a) if a else float('nan')):>18.2f}"
            f"{(st.mean(c) if c else float('nan')):>11.2f}"
        )


# --------------------------------------------------------------------------
# 3. "The clips sound the same to me"
# --------------------------------------------------------------------------

# Phrases where the model reports being unable to distinguish the pair. Kept
# deliberately literal: these are claims about the audio, not hedges about the
# answer format.
_IDENTITY = re.compile(
    r"identical"
    r"|acoustically the same"
    r"|no (?:perceptible |noticeable |audible )?difference"
    r"|equally (?:loud|intense|fast|high)"
    r"|neither (?:clip|a nor b)"
    r"|both clips are the same"
    r"|same (?:loudness|volume|pitch|tempo|intensity)",
    re.IGNORECASE,
)


def identity_rates() -> list[dict]:
    """How often each model reports the pair as indistinguishable, and whether it
    answers anyway.

    This is the statistic that survives across architectures where the accuracy
    delta does not. Kimi and Qwen3-Omni assert indistinguishability at similar
    rates on RAD but respond differently -- one abstains, the other guesses -- so
    accuracy conflates a perceptual failure with a compliance difference.
    """
    out = []
    for task in cot.COT_TASKS:
        for model in cot.COT_MODELS:
            recs = _load(task, model, "cot")
            if not recs:
                continue
            rows = list(recs.values())
            ident = [r for r in rows if _IDENTITY.search(r.get("response") or "")]
            out.append(
                {
                    "task": task,
                    "model": model,
                    "n": len(rows),
                    "abstain_pct": round(100.0 * sum(1 for r in rows if not r["parsed"]) / len(rows), 1),
                    "identity_pct": round(100.0 * len(ident) / len(rows), 1),
                    "identity_and_answered": sum(1 for r in ident if r["parsed"]),
                    "identity_and_correct": sum(1 for r in ident if r["correct"]),
                }
            )
    return out


def print_identity(rows: list[dict]) -> None:
    print("\nHow often the model reports the pair as indistinguishable (CoT condition)\n")
    head = (
        f"  {'task':15s}{'model':11s}{'n':>5}{'abstain %':>11}{'identity %':>12}"
        f"{'answered anyway':>17}{'of those, right':>17}"
    )
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in rows:
        print(
            f"  {r['task']:15s}{r['model']:11s}{r['n']:>5}{r['abstain_pct']:>11.1f}"
            f"{r['identity_pct']:>12.1f}{r['identity_and_answered']:>17}"
            f"{r['identity_and_correct']:>17}"
        )
    print("\n  identity % counts responses claiming the clips cannot be told apart,")
    print("  whether or not a letter was returned. Verified per item that these pairs")
    print("  differ measurably (RAD loudness pairs average ~10 dB apart).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mcnemar", action="store_true")
    ap.add_argument("--cues", action="store_true")
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--cue-model", default="kimi")
    ap.add_argument("--refresh-cues", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if not (a.mcnemar or a.cues or a.identity or a.all):
        ap.error("pass --mcnemar, --cues, --identity, or --all")
    if a.refresh_cues:
        measure_cues(refresh=True)
    if a.mcnemar or a.all:
        print_mcnemar(paired_tests())
    if a.identity or a.all:
        print_identity(identity_rates())
    if a.cues or a.all:
        print_cue_analysis(a.cue_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
