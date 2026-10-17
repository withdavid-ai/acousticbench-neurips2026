# AcousticBench: Measuring Acoustic Perception in Large Audio Language Models

**NeurIPS 2026**

Anirudh Rahul, Arth Vidyarthi, Michael Horgan, Suyash Jaju, Yuhao Zhou

> 🚧 **Code and dataset will be released soon.** Watch or star this repository to be notified when they become available.

## Overview

AcousticBench is a benchmark for evaluating how well Large Audio Language Models (LALMs) perceive non-lexical properties of audio, the kind of information that transcript-based (ASR cascade) systems discard. It contains 3,200 two-choice questions across three task families:

- **Relative Acoustic Discrimination (RAD)**
- **Speech Emotion Recognition (SER)**
- **Relative Speech Quality (RSQ)**

The benchmark draws from 5,800 unique recordings (11.6 hours) collected through single-speaker voice-acting sessions, multi-speaker WebRTC conversations in 17 languages, and synthetic sound generation.

## Release Plan

- [ ] Benchmark dataset
- [ ] Evaluation code for reproducibility

## Citation

If you find AcousticBench useful in your research, please cite:

```bibtex
@inproceedings{rahul2026acousticbench,
  title     = {AcousticBench: Measuring Acoustic Perception in Large Audio Language Models},
  author    = {Rahul, Anirudh and Vidyarthi, Arth and Horgan, Michael and Jaju, Suyash and Zhou, Yuhao},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
