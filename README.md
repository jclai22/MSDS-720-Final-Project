# MSDS 720 Final Project – Spotify Bot vs. Human Analysis

> Detecting bot-like behavior in Spotify user data using logistic regression, multiple regression, and interaction modeling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

This project analyzes Spotify user behavior data to distinguish bot-like accounts from human users. Using a combination of regression techniques and engineered features (skip rate, diversity score), we investigate which behavioral signals predict bot classification and whether bot-like accounts artificially inflate stream counts.

### Research Questions

**RQ1 (Logistic Regression):** Among Spotify users, which behavioral features -- specifically skip rate, listening diversity, listening time, and genre preference -- significantly predict the likelihood that an account exhibits bot-like behavior?

**RQ2 (Multiple Regression + Interaction):** After controlling for user demographics and listening behavior, do accounts classified as bot-like show significantly higher stream counts than human accounts, and does bot-like status moderate the relationship between listening time and total streams?

### Hypotheses

| # | Hypothesis |
|---|---|
| H1a | Higher skip rates are positively associated with the probability of bot-like classification (OR > 1, p < .05) |
| H1b | Lower listening diversity scores are positively associated with the probability of bot-like classification (OR < 1, p < .05) |
| H2 | Bot-like accounts will have significantly higher mean stream counts than human accounts, controlling for listening time and age (beta > 0, p < .05) |
| H2a | There is a significant positive interaction between listening time and bot-like status on stream counts, such that the marginal effect of listening time on streams is greater for bot-like accounts than for human users (beta_interaction > 0, p < .05) |

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

## Environment Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yml
```

This creates an environment called `msds720` with Python 3.11 and all required packages (pandas, numpy, scikit-learn, statsmodels, matplotlib, seaborn, etc.).

### 2. Activate the environment

```bash
conda activate msds720
```

### 3. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name msds720 --display-name "MSDS 720 (Python 3.11)"
```

After this step, select **MSDS 720 (Python 3.11)** as the kernel when opening notebooks in Jupyter or VS Code.

### 4. Alternative: pip install

If you prefer pip over conda:

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
├── paper/
│   ├── literature_review.tex  # LaTeX literature review (Overleaf-ready)
│   └── references.bib         # BibTeX bibliography
├── references/                # Literature and reference PDFs
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Data loading, cleaning, feature engineering
│   └── eda.py                # Reusable EDA plotting functions
├── environment.yml           # Conda environment specification
├── requirements.txt          # Pip requirements (alternative)
├── .gitignore
├── LICENSE
└── README.md
```

## Group Members

- **JC** – Dataset, regression, interaction effects
- **TN** – Logistic regression, model selection, discussion

## License

MIT - See [LICENSE](LICENSE) for details.
