# FMDS Extraction Benchmark

This repository contains the dataset, prompts, evaluation scripts, and human benchmark results accompanying the master's thesis:

> **Automating Injury Surveillance: LLMs for Clinical Data Extraction**
> Håvard Skjærstein, UiT The Arctic University of Norway, May 2026

The thesis investigates whether Large Language Models can automatically extract Felles Minimum Datasett (FMDS) fields from unstructured Norwegian clinical injury notes, with the goal of reducing documentation workload and improving the completeness of national injury data.

## Repository Structure

```
fmds-extraction-benchmark/
├── dataset/
│   ├── fine-tuning/
│   │   ├── cases.json
│   │   └── gold-standard.json
│   └── testing/
│       ├── cases.json
│       └── gold-standard.json
├── evaluation/
│   ├── evaluate-cloud.py
│   └── evaluate-on-premise.py
├── human-benchmark/
│   ├── annotator-1.json
│   ├── annotator-2.json
│   └── gold-standard.json
└── prompts/
    ├── few-shot.json
    └── zero-shot.json
```

## Contents

### Prompts
- `zero-shot.json` — The schema-constrained system prompt used for all zero-shot experiments, defining the FMDS schema, field descriptions, allowed values, and output formatting rules.
- `few-shot.json` — The extended prompt including four representative input-output examples used for few-shot experiments.

### Dataset
- `fine-tuning/cases.json` — 500 synthetic clinical injury cases used for fine-tuning open-weight models.
- `fine-tuning/gold-standard.json` — Gold-standard FMDS annotations for the fine-tuning cases.
- `testing/cases.json` — 310 synthetic clinical injury cases used for evaluation.
- `testing/gold-standard.json` — Gold-standard FMDS annotations for the test cases, validated by a healthcare professional.

All cases are written in Norwegian to reflect the linguistic context of clinical documentation in Norwegian hospitals.

### Evaluation
- `evaluate-cloud.py` — Evaluation script for cloud-hosted frontier model outputs.
- `evaluate-on-premise.py` — Evaluation script for on-premise open-weight model outputs.

Both scripts parse model JSON outputs, normalize field values, and compute exact-match accuracy, average field accuracy, and per-field accuracy against the gold standard.

### Human Benchmark
- `annotator-1.json` / `annotator-2.json` — FMDS annotations completed by emergency department clinicians on the first 50 test cases.
- `gold-standard.json` — The corresponding gold-standard annotations used for comparison.

## Fine-Tuned Models

Fine-tuned open-weight models are available on Hugging Face: [huggingface.co/havaskj](https://huggingface.co/havaskj)

## Citation

If you use this benchmark in your research, please cite:

```
Skjærstein, H. (2026). Automating Injury Surveillance: LLMs for Clinical Data Extraction.
Master's thesis, UiT The Arctic University of Norway.
```

## License

This dataset and code are released for research purposes. Please see the repository license for details.