# Tech Stack Recommender System (Project 3)

## Overview
This builds a Content-Based Recommender System that maps a user's input skills to predefined job roles. By computing the Cosine Similarity between TF-IDF vectorized skill arrays, the system calculates which tech roles match your stack the most accurately.

## Features
- Implements Term Frequency-Inverse Document Frequency (TF-IDF) via `scikit-learn` to vectorize skillsets.
- Custom tokenizer that handles space-separated multi-word skills (like `machine-learning` or `deep-learning`).
- Solves the Cold Start Problem: if the algorithm cannot find any matching skills, it falls back to recommending general trending tech roles.
- Returns a ranked `Top-N` (Top 3) list indicating matching percentage to prevent choice overload.

## Dataset
This relies on an included `raw_skills.csv` listing various job roles (like Data Scientist, DevOps Engineer, Full Stack Developer, etc.) and their associated space-separated skills.

## Prerequisites
- Python 3.x
- `pandas`
- `scikit-learn`

Install the required dependencies:
```bash
pip install pandas scikit-learn
```

## How to Run
```bash
python recommender.py
```

## Example Interaction
```text
Welcome to the Tech Stack Recommender!

Enter your skills (space-separated): python sql aws

Top 3 Recommended Career Paths:
1. Data Engineer        — Match: 58%
2. Data Scientist       — Match: 45%
3. DevOps Engineer      — Match: 32%
```
