# 🎯 Subsify

AI-powered customer subscription prediction platform with advanced machine learning models.

## Features

- **Interactive Web Interface**: User-friendly form for entering customer data
- **Model Selection**: Choose between Naive Bayes and Neural Network algorithms
- **Real-time Predictions**: Instant subscription probability calculations with confidence scores
- **Customer Insights**: Actionable recommendations based on customer behavior patterns
- **Model Transparency**: Clear explanation of prediction factors and algorithm differences
- **Professional Branding**: Clean, modern interface designed for business users

## Model Details

- **Primary Algorithm**: Gaussian Naive Bayes (fast, interpretable)
- **Secondary Algorithm**: Neural Network (complex patterns, higher accuracy potential)
- **Web App**: Both models available with real-time switching
- **Features**: 9 customer attributes (usage patterns, behavior metrics, demographics)
- **Training Data**: 1000 synthetic customer records with identical train/test splits
- **Performance**: Both models optimized for subscription prediction accuracy
- **Validation**: Stratified 80/20 split with consistent random state (42) for reproducibility

## Usage

1. Select your preferred ML model (Naive Bayes or Neural Network)
2. Enter customer data in the form fields
3. Click "Predict Subscription" to get results
4. View subscription probability and confidence score
5. Read customer insights and recommendations
6. Compare results between different models

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

- `streamlit_app.py`: Main Subsify web application
- `naive_bayes_model.py`: Naive Bayes model training script
- `neural_network_model.py`: Neural Network model training script
- `naive_bayes_subscription_model.joblib`: Trained Naive Bayes model
- `naive_bayes_scaler.joblib`: Feature scaler for Naive Bayes
- `neural_network_subscription_model.joblib`: Trained Neural Network model
- `neural_network_scaler.joblib`: Feature scaler for Neural Network
- `requirements.txt`: Python dependencies
- `README.md`: This documentation
