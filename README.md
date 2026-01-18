# Credit Scoring Demo

A proof-of-concept demonstrating automated credit scoring using two independent methods:

1. **ML-based Credit Tier Prediction** - Pre-trained XGBoost model predicts credit tier (Excellent/Good/Fair/Poor) and calculates credit limit
2. **Statistical Credit Score** - Formula-based calculation producing a 300-850 credit score

## What's Included

- `Credit_Scoring_Demo.ipynb` - Interactive notebook explaining both methods with formulas
- `backend/` - Python scripts for credit scoring
  - `simple_credit_score.py` - Standalone script for both methods
  - `stat_score_util.py` - Statistical scoring utilities
  - `model/` - Pre-trained ML models
- `data/` - Sample user data

## Quick Start

### Option 1: Jupyter Notebook (Recommended)

```bash
jupyter notebook Credit_Scoring_Demo.ipynb
```

The notebook demonstrates:
- How the ML model predicts credit tiers
- How the statistical formula calculates credit scores (300-850)
- Detailed formulas and weights used
- Examples with real user data

### Option 2: Python Script

```bash
cd backend
python simple_credit_score.py --user-id 8625
```

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost joblib scipy jupyter
```

Optional (for MongoDB):
```bash
pip install pymongo python-dotenv
```
