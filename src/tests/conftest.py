import pytest
from src.app import app
import pandas as pd

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def legit_series():
    return pd.Series({
        "amount": 100,
        "time": 12,
        "prev_fraud": 0,
        "account_age": 365,
        "transactions_24h": 2,
        "txn_type": "Online Purchase",
        "location": "Local",
        "device": "Desktop"
    })

@pytest.fixture
def fraud_series():
    return pd.Series({
        "amount": 20000,
        "time": 2,
        "prev_fraud": 2,
        "account_age": 10,
        "transactions_24h": 30,
        "txn_type": "ATM Withdrawal",
        "location": "Foreign",
        "device": "Mobile"
    })
@pytest.fixture
def legit_payload():
    return {
        "amount": 100,
        "time": 12,
        "prev_fraud": 0,
        "account_age": 365,
        "transactions_24h": 2,
        "txn_type": "Online Purchase",
        "location": "Local",
        "device": "Desktop"
    }

@pytest.fixture
def fraud_payload():
    return {
        "amount": 20000,
        "time": 2,
        "prev_fraud": 2,
        "account_age": 10,
        "transactions_24h": 30,
        "txn_type": "ATM Withdrawal",
        "location": "Foreign",
        "device": "Mobile"
    }
@pytest.fixture
def moderate_payload():
    return {
        "amount": 5000,
        "time": 10,
        "prev_fraud": 1,
        "account_age": 100,
        "transactions_24h": 10,
        "txn_type": "Online Purchase",
        "location": "Local",
        "device": "Mobile"
    }