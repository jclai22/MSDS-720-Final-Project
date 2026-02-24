# MSDS 720 Final Project – Spotify Bot vs. Human Analysis

This repository contains the code, data, and reference materials for our final project in **MSDS 720 Advanced Statistics** at Meharry Medical College.

## Research Questions

**RQ1 (Logistic Regression):** Which behavioral features predict whether a Spotify account is bot-like or human?

**RQ2 (Multiple Regression + Interaction):** Does bot-like behavior inflate total stream counts, and does the effect of listening time on streams differ between bot-like and human users?

## Hypotheses

- **H1:** Accounts with higher skip rates, lower diversity scores, and more repetitive listening patterns will be more likely classified as bot-like.
- **H2:** Bot-like accounts will have significantly higher stream counts than human accounts.
- **H2a (Interaction):** The positive effect of listening time on streams will be stronger for bot-like accounts than for human users.

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

## Tech Stack

- **Language:** Python
- **Key Libraries:** pandas, numpy, scikit-learn, statsmodels, matplotlib, seaborn

## Project Structure

```
├── LICENSE
├── README.md
├── .gitignore
├── Reference Materials (PDFs)
└── (analysis scripts and notebooks coming soon)
```

## Group Members

- **JC** – Dataset, regression, interaction effects
- **TN** – Logistic regression, model selection, discussion

## License

This project is licensed under the [MIT License](LICENSE).
