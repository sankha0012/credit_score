# Credit Scoring Demo - AI Agent Instructions

## Project Overview

A proof-of-concept credit scoring system with **two independent parallel methods**:

1. **Statistical Credit Score** (300-850 range) - Formula-based, no ML required, always available
2. **ML-Based Credit Tier** (Excellent/Good/Fair/Poor) - XGBoost model, optional, requires dependencies

Both methods operate independently and can be used interchangeably depending on deployment context (lightweight vs. full ML setup).

## Architecture & Data Flow

### Key Components

- **[backend/simple_credit_score.py](backend/simple_credit_score.py)** - Main entry point, unified API for both scoring methods
  - `calculate_statistical_credit_score(user_data)` - Core statistical logic
  - `predict_with_ml(user_data_df)` - Optional ML prediction layer
  - `score_user(user_id, use_ml, use_mongodb)` - Unified interface handling both methods + data source abstraction

- **[backend/stat_score_util.py](backend/stat_score_util.py)** - Legacy statistical scoring (now integrated into main module)

- **[backend/dummy.py](backend/dummy.py)** - `PrepareDummyCols` transformer for categorical encoding before ML prediction

- **[backend/model/](backend/model/)** - Joblib serialized pipeline:
  - `*_le.jlb` - LabelEncoder for output classes
  - `*_coldummy.jlb` - PrepareDummyCols transformer (categorical encoding)
  - `*_ordenc.jlb` - OrdinalEncoder for numeric categoricals
  - `*_model.jlb` - XGBoost classifier

- **[data/](data/)** - JSON and training data
  - `user_data.json` - Customer records (~5700 records, ~425K lines)
  - `train_data.json` - Training set (>50MB, skip for reads)

### Data Flow

```
User Input (Customer_ID)
    ↓
get_user_data() → MongoDB or JSON file
    ↓
calculate_statistical_credit_score() → (300-850 score)
    ├─ Payment history (5%)
    ├─ Credit utilization (50%, max 40% threshold)
    ├─ History length (2.5%, normalized with mean=221.22, std=99.68)
    ├─ Outstanding debt (2.5%, normalized with mean=1426.22, std=1155.13)
    └─ Recent inquiries (40%, normalized with mean=5.80, std=3.87)
    ↓
[Optional] predict_with_ml() → Credit tier + probabilities
    ├─ PrepareDummyCols.transform() - Encode categorical vars
    ├─ OrdinalEncoder.transform() - Encode numeric categoricals
    └─ XGBoost.predict_proba() → [Good, Poor, Standard] probabilities
```

## Critical Patterns & Conventions

### 1. Statistical Score Normalization

**Pattern**: Percentile-based weighting for non-linear scoring factors

```python
# Field values converted to 0-1 percentile using normal distribution
z_score = (value - mean) / std
percentile = 0.5 * (1 + math.erf(z_score / math.sqrt(2)))
```

- **Credit history length**: Uses population mean (221.22) and std (99.68)
- **Outstanding debt**: Uses mean (1426.22) and std (1155.13)
- **Recent inquiries**: Uses mean (5.80) and std (3.87)

⚠️ **Key**: These population statistics are hardcoded in [simple_credit_score.py#L117-L128](backend/simple_credit_score.py#L117-L128). Do NOT change without recalibrating on actual data distribution.

### 2. Credit Utilization Hard Threshold

**Conditional Logic**: If utilization > 40%, score is 0 (automatic disqualification for that factor)

```python
if utilization > 0.4:
    scores["credit_utilization"] = 0
else:
    scores["credit_utilization"] = 1 - utilization
```

This differs from smooth penalization—it's binary. Preserve this behavior in refactors.

### 3. Flexible Data Source

**Pattern**: Abstraction layer with graceful degradation

```python
score_user(user_id, use_ml=False, use_mongodb=False)
    ├─ try MongoDB (requires MONGO_CONNECTION_STRING, MONGODB_DB env vars)
    ├─ fall back to JSON (data/user_data.json)
    └─ Gracefully handle both available and missing dependencies
```

- JSON fallback ensures offline/lightweight deployments work
- ML optional—statistical scoring always runs
- MongoDB optional—JSON is default

### 4. User Data Schema

Critical fields (from [data/user_data.json](data/user_data.json)):

```json
{
  "Customer_ID": int,
  "Name": string,
  "Age": int,
  "Annual_Income": float,
  "Monthly_Inhand_Salary": float,
  "Credit_History_Age": int (months),
  "Num_of_Delayed_Payment": int,
  "Credit_Utilization_Ratio": float (0-100%),
  "Outstanding_Debt": float,
  "Num_Credit_Inquiries": int,
  "Credit_Mix": "Good|Bad|Poor" (string),
  "Type_of_Loan": string,
  "Payment_Behaviour": string,
  ...
}
```

### 5. ML Model Pipeline Order

**Critical**: Exact transform sequence for inference (see [simple_credit_score.py#L176-L180](backend/simple_credit_score.py#L176-L180)):

1. Drop non-numeric IDs and names
2. Apply PrepareDummyCols (handles `","` separated categorical values)
3. Apply OrdinalEncoder (numeric categoricals)
4. Predict with XGBoost using exact feature ordering

⚠️ Feature order matters. Use `model.feature_names_in_` to validate.

## Common Tasks

### Add a New Scoring Factor

1. Add weight to `weights` dict in `calculate_statistical_credit_score()`
2. Calculate percentile if non-linear (use population stats from docstring or recompute)
3. Normalize to 0-1 range
4. Update score tier thresholds (lines 125-133) if needed

### Extend Data Source (e.g., PostgreSQL)

Follow [get_user_from_mongodb()](backend/simple_credit_score.py#L233) pattern:
- Create `get_user_from_postgres()` 
- Add conditional in `get_user_data()`
- Update CLI args in `main()`

### Test New Data

Use [Credit_Scoring_Demo.ipynb](Credit_Scoring_Demo.ipynb) interactive cells—they reload both methods and visualize breakdowns.

Run CLI: `python backend/simple_credit_score.py --user-id 8625 --use-ml`

## Dependencies & Setup

**Minimal** (statistical scoring only):
- Python 3.8+
- No external deps

**Full** (with ML):
```bash
pip install pandas numpy scikit-learn xgboost joblib scipy jupyter
```

**Optional** (MongoDB):
```bash
pip install pymongo python-dotenv
```

Set env vars for MongoDB:
```bash
MONGO_CONNECTION_STRING=mongodb://localhost:27017
MONGODB_DB=credit_scoring
```

## Debugging & Troubleshooting

| Issue | Diagnosis |
|-------|-----------|
| User not found | Check `user_data.json` exists and JSON is valid; verify Customer_ID exists |
| ML model error | Run `--use-ml`; check if joblib models exist in `backend/model/` |
| Wrong score range | Verify weights sum to 1.0; check hardcoded population stats |
| Feature mismatch in ML | Ensure PrepareDummyCols encoded categoricals correctly; check for new data types |

## Files You'll Frequently Edit

- [backend/simple_credit_score.py](backend/simple_credit_score.py) - Scoring logic, weights, thresholds
- [backend/stat_score_util.py](backend/stat_score_util.py) - Legacy utilities (consider consolidating into main module)
- [Credit_Scoring_Demo.ipynb](Credit_Scoring_Demo.ipynb) - Documentation and testing

## Guardrails

- ✅ Statistical scoring must handle missing fields gracefully (use `.get()` with defaults)
- ✅ Score always normalizes to 300-850 range; clamp with `max(300, min(850, score))`
- ✅ Tier classification thresholds are non-negotiable (750→Excellent, 700→Good, etc.)
- ✅ ML model serialization uses joblib; do not try to reload with pickle
- ✅ All user data is treated as untrusted input (validate types before calculations)
