# MSDS 720 Final Project – Spotify Bot vs. Human Analysis

> Detecting bot-like behavior in Spotify user data using logistic regression, multiple regression, and interaction modeling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

This project analyzes Spotify user behavior data to distinguish bot-like accounts from human users. Using a combination of regression techniques and engineered features (skip rate, diversity score), we investigate which behavioral signals predict bot classification and whether bot-like accounts artificially inflate stream counts.

### Research Questions

**RQ1 (Logistic Regression):** Which behavioral features predict whether a Spotify account is bot-like or human?

**RQ2 (Multiple Regression + Interaction):** Does bot-like behavior inflate total stream counts, and does the effect of listening time on streams differ between bot-like and human users?

### Hypotheses

| # | Hypothesis | Expected Direction |
|---|---|---|
| H1 | Skip rate, diversity score, and repetitive listening predict bot classification | Higher skip rate, lower diversity → bot-like |
| H2 | Bot-like accounts have higher stream counts | Bot > Human |
| H2a | Listening time effect on streams differs by account type | Stronger effect for bot-like accounts |

## Dataset

- **Primary:** [Spotify User Behavior Dataset (Kaggle)](https://www.kaggle.com/datasets/meeraajayakumar/spotify-user-behavior-dataset)
- **Secondary:** [Spotify User Behavior Survey Data (Kaggle)](https://www.kaggle.com/datasets/coulsonlll/spotify-user-behavior-survey-data)

## Statistical Methods

| Method | Description |
|---|---|
| Multiple Linear Regression | Predict streams/listening minutes; VIF checks |
| Logistic Regression / GLM | Predict bot-like accounts; odds ratios + 95% CI |
| Interaction Effects | `listening_time x bot_like`; interaction plots |
| Model Selection | AIC/BIC comparison across candidate models |

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.data_loader import load_and_clean_data

df = load_and_clean_data("data/raw/Spotify_data.xlsx")
```

## Tech Stack

- **Language:** Python
- **Key Libraries:** pandas, numpy, scikit-learn, statsmodels, matplotlib, seaborn

## Coding Standards

- **PEP 8 Compliance** — All Python code follows [PEP 8](https://peps.python.org/pep-0008/) style guidelines
- **DRY Principle** — Don't Repeat Yourself; shared logic is extracted into reusable functions and modules
- **Modular Design** — Code is organized into clear, single-responsibility functions
- **Readable Code** — Descriptive variable and function names; comments only where logic isn't self-evident

## Project Structure

```
├── data/
│   ├── raw/                  # Original dataset (not tracked)
│   └── cleaned/              # Cleaned, analysis-ready CSV
├── notebooks/
│   └── 01_eda.ipynb          # Exploratory data analysis
├── references/               # Literature and reference PDFs
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Data loading, cleaning, feature engineering
│   └── eda.py                # Reusable EDA plotting functions
├── .gitignore
├── LICENSE
└── README.md
```

## Group Members

- **JC** – Dataset, regression, interaction effects
- **TN** – Logistic regression, model selection, discussion

## License

MIT - See [LICENSE](LICENSE) for details.
