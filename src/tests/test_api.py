"""
Tests for the /predict Flask endpoint.
Field names match exactly what app.py reads from request.get_json():
  amount, time, prev_fraud, account_age, transactions_24h,
  type, location, device, payment_method
"""
import pytest
import json
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


class TestHomeRoute:

    def test_home_returns_200(self, client):
        response = client.get('/')
        assert response.status_code == 200


class TestPredictBasicBehavior:

    def test_legitimate_transaction_returns_200(self, client, legit_payload):
        response = client.post('/predict',
                               data=json.dumps(legit_payload),
                               content_type='application/json')
        assert response.status_code == 200

    def test_fraudulent_transaction_returns_200(self, client, fraud_payload):
        response = client.post('/predict',
                               data=json.dumps(fraud_payload),
                               content_type='application/json')
        assert response.status_code == 200

    def test_response_status_is_success(self, client, legit_payload):
        response = client.post('/predict',
                               data=json.dumps(legit_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        assert data['status'] == 'success'

    def test_response_has_prediction_field(self, client, legit_payload):
        response = client.post('/predict',
                               data=json.dumps(legit_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        assert 'prediction' in data

    def test_prediction_is_valid_value(self, client, legit_payload):
        response = client.post('/predict',
                               data=json.dumps(legit_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        assert data['prediction'] in ['Legitimate', 'Fraudulent']

    def test_response_has_fraud_probability(self, client, legit_payload):
        response = client.post('/predict',
                               data=json.dumps(legit_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        assert 'fraud_probability' in data

    def test_fraud_probability_is_percentage_string(self, client, legit_payload):
        """Your app returns probability as formatted string like '25.0%'."""
        response = client.post('/predict',
                               data=json.dumps(legit_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        assert data['fraud_probability'].endswith('%')

    def test_response_has_analysis_block(self, client, legit_payload):
        response = client.post('/predict',
                               data=json.dumps(legit_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        assert 'analysis' in data


class TestAnalysisStructure:
    """Tests for the nested analysis object your app returns."""

    def test_analysis_has_transaction_details(self, client, legit_payload):
        response = client.post('/predict',
                               data=json.dumps(legit_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        assert 'transaction_details' in data['analysis']

    def test_analysis_has_risk_analysis(self, client, legit_payload):
        response = client.post('/predict',
                               data=json.dumps(legit_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        assert 'risk_analysis' in data['analysis']

    def test_analysis_has_ml_analysis(self, client, legit_payload):
        response = client.post('/predict',
                               data=json.dumps(legit_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        assert 'ml_analysis' in data['analysis']

    def test_risk_score_format_is_x_of_16(self, client, legit_payload):
        """Your app returns score as string like '4/16'."""
        response = client.post('/predict',
                               data=json.dumps(legit_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        score_str = data['analysis']['risk_analysis']['score']
        assert '/16' in score_str

    def test_risk_level_is_valid(self, client, legit_payload):
        response = client.post('/predict',
                               data=json.dumps(legit_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        level = data['analysis']['risk_analysis']['level']
        assert level in ['Low', 'Medium', 'High']

    def test_ml_analysis_has_probability(self, client, legit_payload):
        response = client.post('/predict',
                               data=json.dumps(legit_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        assert 'probability' in data['analysis']['ml_analysis']

    def test_feature_importance_has_8_features(self, client, legit_payload):
        """Your model uses 8 FEATURE_COLUMNS exactly."""
        response = client.post('/predict',
                               data=json.dumps(legit_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        fi = data['analysis']['ml_analysis']['feature_importance']
        assert len(fi) == 8


class TestPredictionAccuracy:
    """Tests that verify correct predictions matching your screenshots."""

    def test_legitimate_transaction_predicts_legitimate(self, client, legit_payload):
        """ss1/ss2: $500, 2PM, no fraud history, 365 day account → Legitimate."""
        response = client.post('/predict',
                               data=json.dumps(legit_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        assert data['prediction'] == 'Legitimate'

    def test_fraud_transaction_predicts_fraudulent(self, client, fraud_payload):
        """ss5/ss6: $15000, 2AM, foreign, new account → Fraudulent, 100% score."""
        response = client.post('/predict',
                               data=json.dumps(fraud_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        assert data['prediction'] == 'Fraudulent'

    def test_fraud_transaction_risk_score_is_max(self, client, fraud_payload):
        """ss6 screenshot shows 16/16 risk score."""
        response = client.post('/predict',
                               data=json.dumps(fraud_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        assert data['analysis']['risk_analysis']['score'] == '16/16'

    def test_moderate_risk_transaction_flagged(self, client, moderate_payload):
        """ss3/ss4: moderate risk combo → Fraudulent (your system flags it)."""
        response = client.post('/predict',
                               data=json.dumps(moderate_payload),
                               content_type='application/json')
        data = json.loads(response.data)
        assert data['prediction'] == 'Fraudulent'

    def test_fraud_probability_higher_for_fraud(self, client, legit_payload, fraud_payload):
        """Fraudulent transaction must have higher probability than legitimate."""
        r_legit = client.post('/predict', data=json.dumps(legit_payload),
                              content_type='application/json')
        r_fraud = client.post('/predict', data=json.dumps(fraud_payload),
                              content_type='application/json')
        legit_prob = float(json.loads(r_legit.data)['fraud_probability'].strip('%'))
        fraud_prob = float(json.loads(r_fraud.data)['fraud_probability'].strip('%'))
        assert fraud_prob > legit_prob


class TestErrorHandling:

    def test_missing_field_returns_error_status(self, client):
        """Missing required field — app catches exception and returns error status."""
        bad_data = {"amount": "500"}  # Missing all other fields
        response = client.post('/predict',
                               data=json.dumps(bad_data),
                               content_type='application/json')
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'error' in data

    def test_system_does_not_crash_on_bad_input(self, client):
        """Even on bad input, app should return JSON, not crash with 500."""
        response = client.post('/predict',
                               data=json.dumps({"garbage": "data"}),
                               content_type='application/json')
        assert response.status_code == 200  # App catches internally


class TestModelConsistency:

    def test_same_input_gives_same_prediction(self, client, legit_payload):
        """Model must be deterministic — same input = same output."""
        r1 = client.post('/predict', data=json.dumps(legit_payload),
                         content_type='application/json')
        r2 = client.post('/predict', data=json.dumps(legit_payload),
                         content_type='application/json')
        d1 = json.loads(r1.data)
        d2 = json.loads(r2.data)
        assert d1['prediction'] == d2['prediction']
        assert d1['fraud_probability'] == d2['fraud_probability']
