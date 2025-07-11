# 🔮 Subscription Predictor

A machine learning web app that predicts customer subscription likelihood using Naive Bayes algorithm.

## Features

- **Interactive Web Interface**: Easy-to-use form for entering customer data
- **Real-time Predictions**: Instant subscription probability calculations
- **Customer Insights**: Actionable recommendations based on customer behavior
- **Model Transparency**: Clear explanation of prediction factors

## Model Details

- **Algorithm**: Gaussian Naive Bayes
- **Features**: 9 customer attributes (usage patterns, behavior metrics, demographics)
- **Training Data**: 1000 synthetic customer records
- **Performance**: Optimized for subscription prediction accuracy

## Usage

1. Enter customer data in the form fields
2. Click "Predict Subscription" to get results
3. View subscription probability and confidence score
4. Read customer insights and recommendations

## Input Features

### Usage Metrics
- Number of Uses (1-50)
- Average Interval between uses (1-120 days)
- Total Spend ($50-$8000)
- Account Age (30-1095 days)
- Mobile Usage Ratio (0.0-1.0)

### Behavior Metrics
- Support Tickets (0-15)
- Last Login (0-90 days ago)
- Feature Usage Count (1-20)
- Peak Usage Hour (0-23)

## Deployment

This app is designed for easy deployment on Streamlit Cloud:

1. Push this repository to GitHub
2. Connect to Streamlit Cloud
3. Deploy with one click

## Local Development

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Files

- `streamlit_app.py`: Main web application
- `naive_bayes_subscription_model.joblib`: Trained ML model
- `naive_bayes_scaler.joblib`: Feature scaler
- `requirements.txt`: Python dependencies
- `naive_bayes_model.py`: Model training script (for reference)
