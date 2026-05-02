"""
Tests for calculate_risk_score() in app.py.

Your function has 7 rules that sum to a max score of 16:
  - Amount > 10000  → +3 | Amount > 5000 → +2 | Amount > 1000 → +1
  - Time 0–5        → +2
  - Previous fraud  → +3
  - Account age <30 → +2
  - Transactions >20→ +2
  - Foreign location→ +2
  - ATM + Mobile    → +2
"""
import pytest
import sys, os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from src.app import calculate_risk_score


def make_series(**kwargs):
    """Helper: creates a minimal safe transaction, override any field via kwargs."""
    base = {
        'Transaction_Amount': 100.0,
        'Time_of_Transaction': 12,
        'Previous_Fraudulent_Transactions': 0,
        'Account_Age': 365,
        'Number_of_Transactions_Last_24H': 1,
        'Location': 'Local',
        'Transaction_Type': 'Online Purchase',
        'Device_Used': 'Desktop'
    }
    base.update(kwargs)
    return pd.Series(base)


# ── AMOUNT RULES ──────────────────────────────────────────────────────────────

class TestAmountRules:

    def test_low_amount_no_score(self):
        result = calculate_risk_score(make_series(Transaction_Amount=100))
        assert result['score'] == 0
        assert "Very high amount" not in result['reasons']
        assert "High amount" not in result['reasons']
        assert "Moderate amount" not in result['reasons']

    def test_moderate_amount_adds_1(self):
        result = calculate_risk_score(make_series(Transaction_Amount=2000))
        assert result['score'] == 1
        assert "Moderate amount" in result['reasons']

    def test_high_amount_adds_2(self):
        result = calculate_risk_score(make_series(Transaction_Amount=7000))
        assert result['score'] == 2
        assert "High amount" in result['reasons']

    def test_very_high_amount_adds_3(self):
        result = calculate_risk_score(make_series(Transaction_Amount=15000))
        assert result['score'] == 3
        assert "Very high amount" in result['reasons']

    def test_boundary_exactly_10000(self):
        """$10,000 exactly should NOT trigger very high amount (rule is > 10000)."""
        result = calculate_risk_score(make_series(Transaction_Amount=10000))
        assert "Very high amount" not in result['reasons']
        assert result['score'] == 2  # triggers High amount (> 5000)


# ── TIME RULE ─────────────────────────────────────────────────────────────────

class TestTimeRule:

    def test_daytime_no_score(self):
        result = calculate_risk_score(make_series(Time_of_Transaction=14))
        assert result['score'] == 0

    def test_midnight_adds_2(self):
        result = calculate_risk_score(make_series(Time_of_Transaction=0))
        assert result['score'] == 2
        assert "Late night transaction (12AM-5AM)" in result['reasons']

    def test_3am_adds_2(self):
        result = calculate_risk_score(make_series(Time_of_Transaction=3))
        assert result['score'] == 2
        assert "Late night transaction (12AM-5AM)" in result['reasons']

    def test_5am_boundary_adds_2(self):
        """Hour 5 is included (0 <= hour <= 5)."""
        result = calculate_risk_score(make_series(Time_of_Transaction=5))
        assert result['score'] == 2

    def test_6am_no_score(self):
        """Hour 6 is outside the risky window."""
        result = calculate_risk_score(make_series(Time_of_Transaction=6))
        assert result['score'] == 0


# ── PREVIOUS FRAUD RULE ───────────────────────────────────────────────────────

class TestPreviousFraudRule:

    def test_no_previous_fraud_no_score(self):
        result = calculate_risk_score(make_series(Previous_Fraudulent_Transactions=0))
        assert result['score'] == 0

    def test_one_previous_fraud_adds_3(self):
        result = calculate_risk_score(make_series(Previous_Fraudulent_Transactions=1))
        assert result['score'] == 3
        assert "Previous fraud history" in result['reasons']

    def test_multiple_previous_frauds_still_adds_3(self):
        """Score is fixed at +3 regardless of count."""
        result = calculate_risk_score(make_series(Previous_Fraudulent_Transactions=5))
        assert result['score'] == 3


# ── ACCOUNT AGE RULE ──────────────────────────────────────────────────────────

class TestAccountAgeRule:

    def test_old_account_no_score(self):
        result = calculate_risk_score(make_series(Account_Age=365))
        assert result['score'] == 0

    def test_new_account_adds_2(self):
        result = calculate_risk_score(make_series(Account_Age=15))
        assert result['score'] == 2
        assert "New account (<30 days)" in result['reasons']

    def test_boundary_29_days_triggers(self):
        """29 days < 30 — should trigger."""
        result = calculate_risk_score(make_series(Account_Age=29))
        assert result['score'] == 2

    def test_boundary_exactly_30_days_safe(self):
        """Exactly 30 days — should NOT trigger (rule is < 30)."""
        result = calculate_risk_score(make_series(Account_Age=30))
        assert "New account (<30 days)" not in result['reasons']


# ── TRANSACTION FREQUENCY RULE ────────────────────────────────────────────────

class TestFrequencyRule:

    def test_normal_frequency_no_score(self):
        result = calculate_risk_score(make_series(Number_of_Transactions_Last_24H=5))
        assert result['score'] == 0

    def test_high_frequency_adds_2(self):
        result = calculate_risk_score(make_series(Number_of_Transactions_Last_24H=25))
        assert result['score'] == 2
        assert "High transaction frequency" in result['reasons']

    def test_boundary_exactly_20_no_score(self):
        """Exactly 20 should NOT trigger (rule is > 20)."""
        result = calculate_risk_score(make_series(Number_of_Transactions_Last_24H=20))
        assert "High transaction frequency" not in result['reasons']

    def test_boundary_21_triggers(self):
        result = calculate_risk_score(make_series(Number_of_Transactions_Last_24H=21))
        assert result['score'] == 2


# ── LOCATION RULE ─────────────────────────────────────────────────────────────

class TestLocationRule:

    def test_local_no_score(self):
        result = calculate_risk_score(make_series(Location='Local'))
        assert result['score'] == 0

    def test_foreign_adds_2(self):
        result = calculate_risk_score(make_series(Location='Foreign'))
        assert result['score'] == 2
        assert "Foreign location" in result['reasons']


# ── MOBILE ATM RULE ───────────────────────────────────────────────────────────

class TestMobileATMRule:

    def test_atm_with_mobile_adds_2(self):
        result = calculate_risk_score(make_series(
            Transaction_Type='ATM Withdrawal',
            Device_Used='Mobile'
        ))
        assert result['score'] == 2
        assert "Mobile ATM access" in result['reasons']

    def test_atm_with_desktop_no_score(self):
        result = calculate_risk_score(make_series(
            Transaction_Type='ATM Withdrawal',
            Device_Used='Desktop'
        ))
        assert "Mobile ATM access" not in result['reasons']

    def test_online_purchase_with_mobile_no_score(self):
        """Rule only triggers when BOTH ATM + Mobile."""
        result = calculate_risk_score(make_series(
            Transaction_Type='Online Purchase',
            Device_Used='Mobile'
        ))
        assert "Mobile ATM access" not in result['reasons']


# ── RISK LEVEL & COMPOSITE SCORE ─────────────────────────────────────────────

class TestCompositeScore:

    def test_max_score_is_16(self, fraud_series):
        result = calculate_risk_score(fraud_series)
        assert result['max_score'] == 16

    def test_score_never_exceeds_max(self, fraud_series):
        result = calculate_risk_score(fraud_series)
        assert result['score'] <= result['max_score']

    def test_clean_transaction_is_low_risk(self, legit_series):
        result = calculate_risk_score(legit_series)
        assert result['risk_level'] == 'Low'
        assert result['score'] < 4

    def test_suspicious_transaction_is_high_risk(self, fraud_series):
        result = calculate_risk_score(fraud_series)
        assert result['risk_level'] == 'High'
        assert result['score'] >= 8

    def test_screenshot_scenario_score_16(self, fraud_series):
        """Reproduces your ss6 screenshot: $15000, 2AM, prev_fraud=2,
        account_age=15, 25 transactions, Foreign, ATM+Mobile → 16/16."""
        result = calculate_risk_score(fraud_series)
        assert result['score'] == 16
        assert result['risk_level'] == 'High'

    def test_result_has_required_keys(self, legit_series):
        result = calculate_risk_score(legit_series)
        assert 'score' in result
        assert 'max_score' in result
        assert 'risk_level' in result
        assert 'reasons' in result
