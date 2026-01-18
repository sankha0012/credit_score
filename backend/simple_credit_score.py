#!/usr/bin/env python3
"""
Simple Credit Scoring PoC

This script provides two methods for credit scoring:
1. Statistical scorecard (no ML model required)
2. ML-based prediction (optional, requires trained models)

Usage:
    python simple_credit_score.py [--user-id USER_ID] [--use-ml]
    
    Or import and use the functions directly.
"""

import json
import argparse
from pathlib import Path

# Optional: ML model support
USE_ML_MODEL = False
try:
    import pandas as pd
    import numpy as np
    import joblib
    from dummy import PrepareDummyCols
    import __main__
    if not hasattr(__main__, 'PrepareDummyCols'):
        __main__.PrepareDummyCols = PrepareDummyCols
    USE_ML_MODEL = True
except ImportError:
    print("Note: ML dependencies not available. Using statistical scoring only.")

# Optional: MongoDB support
USE_MONGODB = False
try:
    from pymongo import MongoClient
    import os
    from dotenv import load_dotenv
    load_dotenv()
    USE_MONGODB = True
except ImportError:
    print("Note: MongoDB not available. Using JSON files for data.")


# ============================================================================
# STATISTICAL CREDIT SCORING (No ML Required)
# ============================================================================

def calculate_percentile(value, mean, std):
    """Calculate percentile using normal distribution."""
    import math
    z_score = (value - mean) / std
    # Approximate CDF using error function
    percentile = 0.5 * (1 + math.erf(z_score / math.sqrt(2)))
    return percentile


def calculate_statistical_credit_score(user_data):
    """
    Calculate credit score using a weighted statistical scorecard.
    No ML model required.
    
    Args:
        user_data: dict with user financial data
        
    Returns:
        tuple: (score, score_breakdown)
    """
    # Weights for different factors
    weights = {
        "payment_history": 0.05,
        "credit_utilization": 0.50,
        "credit_history_length": 0.025,
        "recent_inquiries": 0.40,
        "outstanding_debt": 0.025,
    }
    
    # Calculate component scores
    scores = {}
    
    # Payment history (higher is better)
    credit_history_age = user_data.get("Credit_History_Age", 220)
    delayed_payments = user_data.get("Num_of_Delayed_Payment", 0)
    if credit_history_age > 0:
        scores["payment_history"] = (credit_history_age - delayed_payments) / credit_history_age
    else:
        scores["payment_history"] = 1.0
    
    # Credit utilization (lower is better, penalize >40%)
    utilization = user_data.get("Credit_Utilization_Ratio", 30) / 100
    if utilization > 0.4:
        scores["credit_utilization"] = 0
    else:
        scores["credit_utilization"] = 1 - utilization
    
    # Credit history length (longer is better)
    # Mean: 221.22, Std: 99.68 (from data analysis)
    scores["credit_history_length"] = calculate_percentile(credit_history_age, 221.22, 99.68)
    
    # Outstanding debt (lower is better)
    # Mean: 1426.22, Std: 1155.13
    outstanding = user_data.get("Outstanding_Debt", 500)
    scores["outstanding_debt"] = 1 - calculate_percentile(outstanding, 1426.22, 1155.13)
    
    # Recent inquiries (fewer is better)
    # Mean: 5.80, Std: 3.87
    inquiries = user_data.get("Num_Credit_Inquiries", 3)
    inquiry_percentile = calculate_percentile(inquiries, 5.80, 3.87)
    if inquiry_percentile > 0.8:
        scores["recent_inquiries"] = 0
    else:
        scores["recent_inquiries"] = 1 - inquiry_percentile
    
    # Calculate weighted score
    weighted_sum = sum(scores[k] * weights[k] for k in weights)
    
    # Normalize to 300-850 range (standard credit score range)
    final_score = int(weighted_sum * 550 + 300)
    final_score = max(300, min(850, final_score))
    
    # Determine credit tier
    if final_score >= 750:
        tier = "Excellent"
    elif final_score >= 700:
        tier = "Good"
    elif final_score >= 650:
        tier = "Fair"
    elif final_score >= 550:
        tier = "Poor"
    else:
        tier = "Very Poor"
    
    return final_score, tier, scores


# ============================================================================
# ML-BASED CREDIT SCORING (Optional)
# ============================================================================

_label_encoder = None
_dummy_transformer = None
_model = None
_ordinal_encoder = None


def load_ml_models():
    """Load ML models lazily."""
    global _label_encoder, _dummy_transformer, _model, _ordinal_encoder
    
    model_dir = Path(__file__).parent / "model"
    
    if _label_encoder is None:
        _label_encoder = joblib.load(model_dir / "credit_score_mul_lable_le.jlb")
    if _dummy_transformer is None:
        _dummy_transformer = joblib.load(model_dir / "credit_score_mul_lable_coldummy.jlb")
    if _model is None:
        _model = joblib.load(model_dir / "credit_score_mul_lable_model.jlb")
    if _ordinal_encoder is None:
        _ordinal_encoder = joblib.load(model_dir / "credit_score_mul_lable_ordenc.jlb")
    
    return _label_encoder, _dummy_transformer, _model, _ordinal_encoder


def predict_with_ml(user_data_df):
    """
    Predict credit score using the pre-trained ML model.
    
    Args:
        user_data_df: pandas DataFrame with user data
        
    Returns:
        tuple: (prediction, probabilities)
    """
    if not USE_ML_MODEL:
        raise RuntimeError("ML dependencies not available. Install pandas, numpy, joblib, scikit-learn.")
    
    label_encoder, dummy, model, ordinal_enc = load_ml_models()
    
    df = user_data_df.copy()
    df.drop(columns=["ID", "Customer_ID", "Name", "SSN", "Credit_Score"], inplace=True, errors='ignore')
    
    df = dummy.transform(df)
    df[ordinal_enc.feature_names_in_] = ordinal_enc.transform(df[ordinal_enc.feature_names_in_])
    
    probabilities = model.predict_proba(df[model.feature_names_in_])[0]
    prediction = label_encoder.inverse_transform(model.predict(df[model.feature_names_in_]))[0]
    
    return prediction, probabilities


# ============================================================================
# DATA ACCESS
# ============================================================================

def get_user_from_json(user_id, data_file=None):
    """Load user data from JSON file."""
    if data_file is None:
        data_file = Path(__file__).parent.parent / "data" / "user_data.json"
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    for record in data:
        if record.get("Customer_ID") == user_id:
            return record
    
    return None


def get_user_from_mongodb(user_id):
    """Load user data from MongoDB."""
    if not USE_MONGODB:
        raise RuntimeError("MongoDB not available. Install pymongo and python-dotenv.")
    
    mongo_conn = os.environ.get("MONGO_CONNECTION_STRING", "mongodb://localhost:27017")
    db_name = os.environ.get("MONGODB_DB", "credit_scoring")
    
    client = MongoClient(mongo_conn)
    col = client[db_name]["user_data"]
    
    user_records = list(col.find({"Customer_ID": int(user_id)}, {"_id": 0}))
    
    if user_records:
        return user_records[0]
    return None


def get_user_data(user_id, use_mongodb=False):
    """Get user data from MongoDB or JSON file."""
    if use_mongodb and USE_MONGODB:
        return get_user_from_mongodb(user_id)
    else:
        return get_user_from_json(user_id)


# ============================================================================
# MAIN SCORING FUNCTION
# ============================================================================

def score_user(user_id, use_ml=False, use_mongodb=False):
    """
    Calculate credit score for a user.
    
    Args:
        user_id: Customer ID
        use_ml: If True, use ML model (requires dependencies)
        use_mongodb: If True, use MongoDB (requires running instance)
        
    Returns:
        dict with scoring results
    """
    # Get user data
    user_data = get_user_data(user_id, use_mongodb=use_mongodb)
    
    if not user_data:
        return {"error": f"User {user_id} not found"}
    
    result = {
        "user_id": user_id,
        "name": user_data.get("Name", "Unknown"),
        "occupation": user_data.get("Occupation", "Unknown"),
        "annual_income": user_data.get("Annual_Income", 0),
    }
    
    # Statistical scoring (always available)
    stat_score, tier, breakdown = calculate_statistical_credit_score(user_data)
    result["statistical_score"] = {
        "score": stat_score,
        "tier": tier,
        "breakdown": breakdown
    }
    
    # ML scoring (optional)
    if use_ml and USE_ML_MODEL:
        try:
            df = pd.DataFrame([user_data])
            prediction, probabilities = predict_with_ml(df)
            
            # Calculate allowed credit limit based on ML prediction
            monthly_income = user_data.get("Monthly_Inhand_Salary", 0)
            # probabilities: [Good, Poor, Standard] typically
            credit_multiplier = 1 * probabilities[0] + 0.5 * probabilities[1] + 0.25 * probabilities[2]
            allowed_credit = int(monthly_income * 6 * credit_multiplier)
            
            result["ml_prediction"] = {
                "credit_tier": prediction,
                "probabilities": {
                    "Good": round(probabilities[0], 4),
                    "Poor": round(probabilities[1], 4),
                    "Standard": round(probabilities[2], 4),
                },
                "allowed_credit_limit": allowed_credit
            }
        except Exception as e:
            result["ml_prediction"] = {"error": str(e)}
    
    return result


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Simple Credit Scoring PoC")
    parser.add_argument("--user-id", type=int, default=3392, help="Customer ID to score")
    parser.add_argument("--use-ml", action="store_true", help="Use ML model for prediction")
    parser.add_argument("--use-mongodb", action="store_true", help="Use MongoDB instead of JSON files")
    parser.add_argument("--list-users", action="store_true", help="List available user IDs")
    
    args = parser.parse_args()
    
    if args.list_users:
        data_file = Path(__file__).parent.parent / "data" / "user_data.json"
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        user_ids = sorted(set(r.get("Customer_ID") for r in data[:100]))
        print(f"Sample user IDs (first 100 unique): {user_ids[:20]}...")
        return
    
    print(f"\n{'='*60}")
    print(f"CREDIT SCORING - User ID: {args.user_id}")
    print(f"{'='*60}")
    
    result = score_user(args.user_id, use_ml=args.use_ml, use_mongodb=args.use_mongodb)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    print(f"\nUser: {result['name']}")
    print(f"Occupation: {result['occupation']}")
    print(f"Annual Income: ${result['annual_income']:,.2f}")
    
    stat = result["statistical_score"]
    print(f"\n--- Statistical Credit Score ---")
    print(f"Score: {stat['score']} ({stat['tier']})")
    print(f"Breakdown:")
    for k, v in stat["breakdown"].items():
        print(f"  - {k}: {v:.3f}")
    
    if "ml_prediction" in result:
        ml = result["ml_prediction"]
        if "error" in ml:
            print(f"\n--- ML Prediction ---")
            print(f"Error: {ml['error']}")
        else:
            print(f"\n--- ML Prediction ---")
            print(f"Credit Tier: {ml['credit_tier']}")
            print(f"Probabilities: {ml['probabilities']}")
            print(f"Allowed Credit Limit: ${ml['allowed_credit_limit']:,}")
    
    print()


if __name__ == "__main__":
    main()
