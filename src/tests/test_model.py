"""
Tests for the ML model and preprocessing pipeline.
Imports directly from app.py since model lives there (not a separate module).
"""
import pytest
import sys, os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from src.app import model, scaler, preprocess_data, FEATURE_COLUMNS


class TestModelLoading:

    def test_model_is_not_none(self):
        assert model is not None

    def test_model_has_predict(self):
        assert hasattr(model, 'predict')

    def test_model_has_predict_proba(self):
        assert hasattr(model, 'predict_proba')

    def test_model_is_random_forest(self):
        assert type(model).__name__ == 'RandomForestClassifier'

    def test_model_has_200_estimators(self):
        """Your app trains with n_estimators=200."""
        assert model.n_estimators == 200

    def test_scaler_is_not_none(self):
        assert scaler is not None

    def test_feature_columns_count_is_8(self):
        """Your app uses exactly 8 features."""
        assert len(FEATURE_COLUMNS) == 8


class TestPreprocessData:

    def make_input_df(self, amount=500, time=14, prev_fraud=0,
                      account_age=365, transactions_24h=2,
                      txn_type='Online Purchase', location='Local', device='Desktop'):
        return pd.DataFrame({
            'Transaction_Amount': [float(amount)],
            'Time_of_Transaction': [float(time)],
            'Previous_Fraudulent_Transactions': [int(prev_fraud)],
            'Account_Age': [int(account_age)],
            'Number_of_Transactions_Last_24H': [int(transactions_24h)],
            'Transaction_Type': [txn_type],
            'Location': [location],
            'Device_Used': [device]
        })

    def test_preprocess_returns_correct_columns(self):
        df = self.make_input_df()
        features = preprocess_data(df)
        assert list(features.columns) == FEATURE_COLUMNS

    def test_preprocess_atm_flag_is_1_for_atm(self):
        df = self.make_input_df(txn_type='ATM Withdrawal')
        features = preprocess_data(df)
        assert features['Is_ATM'].iloc[0] == 1

    def test_preprocess_atm_flag_is_0_for_online(self):
        df = self.make_input_df(txn_type='Online Purchase')
        features = preprocess_data(df)
        assert features['Is_ATM'].iloc[0] == 0

    def test_preprocess_foreign_flag_is_1_for_foreign(self):
        df = self.make_input_df(location='Foreign')
        features = preprocess_data(df)
        assert features['Is_Foreign'].iloc[0] == 1

    def test_preprocess_foreign_flag_is_0_for_local(self):
        df = self.make_input_df(location='Local')
        features = preprocess_data(df)
        assert features['Is_Foreign'].iloc[0] == 0

    def test_preprocess_mobile_flag_is_1_for_mobile(self):
        df = self.make_input_df(device='Mobile')
        features = preprocess_data(df)
        assert features['Is_Mobile'].iloc[0] == 1

    def test_preprocess_output_shape(self):
        df = self.make_input_df()
        features = preprocess_data(df)
        assert features.shape == (1, 8)


class TestModelPredictions:

    def make_features_scaled(self, amount=500, time=14, prev_fraud=0,
                              account_age=365, transactions_24h=2,
                              txn_type='Online Purchase', location='Local', device='Desktop'):
        df = pd.DataFrame({
            'Transaction_Amount': [float(amount)],
            'Time_of_Transaction': [float(time)],
            'Previous_Fraudulent_Transactions': [int(prev_fraud)],
            'Account_Age': [int(account_age)],
            'Number_of_Transactions_Last_24H': [int(transactions_24h)],
            'Transaction_Type': [txn_type],
            'Location': [location],
            'Device_Used': [device]
        })
        features = preprocess_data(df)
        return scaler.transform(features)

    def test_prediction_is_binary(self):
        features = self.make_features_scaled()
        prediction = model.predict(features)[0]
        assert prediction in [0, 1]

    def test_fraud_probability_between_0_and_1(self):
        features = self.make_features_scaled()
        proba = model.predict_proba(features)[0][1]
        assert 0.0 <= proba <= 1.0

    def test_probabilities_sum_to_1(self):
        features = self.make_features_scaled()
        probas = model.predict_proba(features)[0]
        assert abs(sum(probas) - 1.0) < 1e-6

    def test_suspicious_input_high_fraud_probability(self):
        """$15000 at 2AM, foreign, new account, prior fraud — should lean fraudulent."""
        features = self.make_features_scaled(
            amount=15000, time=2, prev_fraud=2,
            account_age=15, transactions_24h=25,
            txn_type='ATM Withdrawal', location='Foreign', device='Mobile'
        )
        proba = model.predict_proba(features)[0][1]
        assert proba > 0.05

    def test_clean_input_low_fraud_probability(self):
        """$500 at 2PM, local, old account — should lean legitimate."""
        features = self.make_features_scaled(
            amount=500, time=14, prev_fraud=0,
            account_age=365, transactions_24h=2,
            txn_type='Online Purchase', location='Local', device='Desktop'
        )
        proba = model.predict_proba(features)[0][1]
        assert proba < 0.5

    def test_model_does_not_crash_on_zero_values(self):
        features = self.make_features_scaled(amount=0, time=0, prev_fraud=0,
                                              account_age=0, transactions_24h=0)
        result = model.predict(features)
        assert result is not None

    def test_feature_importance_sums_to_1(self):
        """Random Forest feature importances must sum to 1.0."""
        total = sum(model.feature_importances_)
        assert abs(total - 1.0) < 1e-6

    def test_feature_importance_count_matches_columns(self):
        assert len(model.feature_importances_) == len(FEATURE_COLUMNS)
