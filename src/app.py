import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='../templates')

# Load dataset
data = pd.read_csv('data/Fraud Detection Dataset.csv')

# Feature columns
FEATURE_COLUMNS = [
    'Transaction_Amount',
    'Time_of_Transaction',
    'Previous_Fraudulent_Transactions',
    'Account_Age',
    'Number_of_Transactions_Last_24H',
    'Is_ATM',
    'Is_Foreign',
    'Is_Mobile'
]

# =========================
# ✅ FIXED RISK SCORE FUNCTION
# =========================
def calculate_risk_score(transaction):
    score = 0
    reasons = []

    # Support BOTH formats
    amount = transaction.get('amount', transaction.get('Transaction_Amount', 0))
    time = transaction.get('time', transaction.get('Time_of_Transaction', 0))
    prev_fraud = transaction.get('prev_fraud', transaction.get('Previous_Fraudulent_Transactions', 0))
    account_age = transaction.get('account_age', transaction.get('Account_Age', 0))
    transactions = transaction.get('transactions_24h', transaction.get('Number_of_Transactions_Last_24H', 0))
    txn_type = transaction.get('txn_type', transaction.get('Transaction_Type', ''))
    location = transaction.get('location', transaction.get('Location', ''))
    device = transaction.get('device', transaction.get('Device_Used', ''))

    # 💰 Amount rules
    if amount > 10000:
        score += 3
        reasons.append("Very high amount")
    elif amount > 5000:
        score += 2
        reasons.append("High amount")
    elif amount > 1000:
        score += 1
        reasons.append("Moderate amount")

    # 🌙 Time rule
    if 0 <= time <= 5:
        score += 2
        reasons.append("Late night transaction (12AM-5AM)")

    # ⚠ Previous fraud
    if prev_fraud > 0:
        score += 3
        reasons.append("Previous fraud history")

    # 🆕 Account age
    if account_age < 30:
        score += 2
        reasons.append("New account (<30 days)")

    # 🔁 Frequency
    if transactions > 20:
        score += 2
        reasons.append("High transaction frequency")

    # 🌍 Location
    if location == "Foreign":
        score += 2
        reasons.append("Foreign location")

    # 📱 ATM + Mobile
    if txn_type == "ATM Withdrawal" and device == "Mobile":
        score += 2
        reasons.append("Mobile ATM access")

    risk_level = "High" if score >= 8 else "Medium" if score >= 4 else "Low"

    return {
        "score": score,
        "max_score": 16,
        "risk_level": risk_level,
        "reasons": reasons
    }

# =========================
# PREPROCESS
# =========================
def preprocess_data(df):
    features = pd.DataFrame()

    features['Transaction_Amount'] = df['Transaction_Amount']
    features['Time_of_Transaction'] = df['Time_of_Transaction']
    features['Previous_Fraudulent_Transactions'] = df['Previous_Fraudulent_Transactions']
    features['Account_Age'] = df['Account_Age']
    features['Number_of_Transactions_Last_24H'] = df['Number_of_Transactions_Last_24H']

    features['Is_ATM'] = (df['Transaction_Type'] == 'ATM Withdrawal').astype(int)
    features['Is_Foreign'] = (df['Location'] == 'Foreign').astype(int)
    features['Is_Mobile'] = (df['Device_Used'] == 'Mobile').astype(int)

    return features[FEATURE_COLUMNS]

# =========================
# TRAIN MODEL
# =========================
X = preprocess_data(data)
y = data['Fraudulent']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    class_weight={0: 1, 1: 3},
    random_state=42
)
model.fit(X_scaled, y)

# =========================
# ROUTES
# =========================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # FIXED: use txn_type instead of type
        input_df = pd.DataFrame({
            'Transaction_Amount': [float(data['amount'])],
            'Time_of_Transaction': [float(data['time'])],
            'Previous_Fraudulent_Transactions': [int(data['prev_fraud'])],
            'Account_Age': [int(data['account_age'])],
            'Number_of_Transactions_Last_24H': [int(data['transactions_24h'])],
            'Transaction_Type': [data['txn_type']],
            'Location': [data['location']],
            'Device_Used': [data['device']]
        })

        features = preprocess_data(input_df)
        features_scaled = scaler.transform(features)

        pred = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0][1]

        # Risk score uses original payload
        risk = calculate_risk_score(data)

        final_prediction = "Fraudulent" if (pred == 1 or risk['score'] >= 4) else "Legitimate"

        final_probability = max(prob, risk['score'] / risk['max_score'])

        response = {
            "status": "success",
            "prediction": final_prediction,
            "fraud_probability": f"{final_probability*100:.1f}%",
            "analysis": {
                "transaction_details": data,
                "risk_analysis": {
                    "score": f"{risk['score']}/{risk['max_score']}",
                    "level": risk['risk_level']
                },
                "ml_analysis": {
                    "probability": prob,
                    "feature_importance": list(model.feature_importances_)
                }
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        })

# =========================
# RUN
# =========================
if __name__ == '__main__':
    app.run(debug=True)