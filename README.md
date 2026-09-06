# AI Legal Contract Clause Classification & Explanation Platform

A small, end-to-end AI application that classifies legal contract clauses into
categories (e.g. Termination, Confidentiality, Governing Law) using a
fine-tuned transformer model, serves predictions through a FastAPI backend,
and provides a Streamlit frontend. Optionally uses the Gemini API to generate
a plain-English explanation of a predicted clause.

This project was built as a learning project to practice practical software
engineering, data engineering, Git, Docker, CI/CD, deployment, logging, and
monitoring around a real AI application — not as a large-scale production
system.

## Status

Work in progress. See `LEARNING_GUIDE.md` for a stage-by-stage walkthrough of
how this project was built, and `ARCHITECTURE.md` for the system design.

## Dataset

Uses the CUAD (Contract Understanding Atticus Dataset), a public dataset of
legal contracts with expert-labeled clauses. Raw data is not committed to
this repository (see `data/README.md` for how to obtain it).

## Project layout

```
data/            raw and processed datasets (raw/ is never modified directly)
models/          trained model artifacts (not committed — see .gitignore)
notebooks/       exploratory notebooks
src/data/        data loading, validation, preprocessing
src/training/    model training and evaluation
src/inference/   prediction logic
src/api/         FastAPI application
src/utils/       shared utilities (logging, etc.)
frontend/        Streamlit UI
tests/           automated tests
.github/workflows/  CI pipeline
```

