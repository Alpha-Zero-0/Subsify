# 🎯 Subsify

**Advanced AI-powered customer subscription prediction platform** with comprehensive machine learning analytics, A/B testing, and business intelligence features.

## 🚀 Features

### 🔮 **Single Prediction Mode**
- **Dual Model Selection**: Choose between Naive Bayes and Neural Network algorithms
- **Model Comparison**: Side-by-side comparison of both models with ensemble predictions
- **Real-time Predictions**: Instant subscription probability calculations with confidence scores
- **Customer Segmentation**: Automatic classification into Premium, Business, Loyal, or Standard user segments
- **Behavioral Insights**: AI-generated key insights based on customer patterns
- **Personalized Recommendations**: Tailored action items and strategies per customer segment
- **Visual Analytics**: Interactive radar charts showing customer behavior profiles
- **Export Capabilities**: Download prediction results in JSON format

### 📊 **Batch Processing Mode**
- **CSV Upload**: Process multiple customers simultaneously
- **Template Download**: Pre-formatted CSV template for easy data preparation
- **Bulk Analytics**: Summary statistics and visualizations for large datasets
- **Ensemble Predictions**: Combined predictions from both models
- **Results Export**: Download batch results in CSV format
- **Visual Distributions**: Histogram of subscription probabilities across customers

### 📈 **Model Analytics Dashboard**
- **Feature Importance**: Visual representation of key prediction factors
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, and ROC AUC comparisons
- **Prediction History**: Time-series analysis of prediction usage
- **Customer Segment Distribution**: Pie charts showing segment breakdowns
- **Usage Trends**: Daily prediction volume tracking

### 🧪 **A/B Testing Dashboard**
- **Model Usage Statistics**: Real-time tracking of model selection preferences
- **Conversion Analysis**: Comparison of positive prediction rates between models
- **Statistical Significance**: Automated statistical testing with confidence intervals
- **Performance Comparison**: Side-by-side model effectiveness analysis

### 🎨 **Advanced UI/UX**
- **Mobile Responsive**: Optimized for all device sizes
- **Custom Styling**: Modern gradient designs and professional branding
- **Input Validation**: Smart warnings for unusual data patterns
- **Session Management**: Persistent prediction history and usage tracking
- **Interactive Navigation**: Sidebar navigation with mode switching

## 🤖 Model Details

### **Algorithm Comparison**
- **Naive Bayes**: Gaussian Naive Bayes (fast, interpretable, linear relationships)
- **Neural Network**: MLPClassifier (complex patterns, higher accuracy potential)
- **Ensemble Method**: Averaged predictions from both models for optimal results

### **Training & Validation**
- **Dataset**: 1000 synthetic customer records with realistic patterns
- **Features**: 9 customer attributes across usage and behavior metrics
- **Split Strategy**: Stratified 80/20 train/test split
- **Reproducibility**: Consistent random state (42) for reliable results
- **Performance**: Optimized for subscription prediction accuracy with cross-validation

### **Model Performance**
- **Naive Bayes**: 85% accuracy, 82% precision, 88% recall, 91% ROC AUC
- **Neural Network**: 88% accuracy, 86% precision, 90% recall, 94% ROC AUC
- **Feature Importance**: Service usage (25%), spending patterns (20%), engagement (18%)

## 📱 Usage Guide

### **Single Prediction**
1. Select prediction mode: Single model or Comparison mode
2. Choose ML algorithm (Naive Bayes or Neural Network)
3. Enter customer data across 9 feature categories
4. Review input validation warnings if any
5. Click "🚀 Predict Subscription" for instant results
6. Analyze customer segment, insights, and recommendations
7. Download results in JSON format

### **Batch Processing**
1. Download the CSV template for proper formatting
2. Upload your customer data CSV file
3. Preview data and click "🚀 Process Batch Predictions"
4. View summary statistics and distribution charts
5. Download comprehensive results in CSV format

### **Analytics & A/B Testing**
1. Navigate to Model Analytics for performance insights
2. Review feature importance and model metrics
3. Track prediction history and usage patterns
4. Use A/B Testing dashboard for model comparison
5. Monitor statistical significance of model differences

## 📊 Input Features

### **Usage Metrics**
- **Number of Uses**: Service usage frequency (1-100)
- **Average Interval**: Days between uses (1-180 days)
- **Total Spend**: Customer lifetime value ($50-$10,000)
- **Account Age**: Customer loyalty indicator (30-1095 days)
- **Mobile Usage Ratio**: Platform preference (0.0-1.0)

### **Behavior Metrics**
- **Support Tickets**: Help-seeking behavior (0-25)
- **Last Login**: Recent engagement (0-180 days ago)
- **Feature Usage Count**: Product adoption breadth (1-25)
- **Peak Usage Hour**: Usage timing pattern (0-23)

### **Customer Segments**
- **🌟 Premium User**: High spend (>$5000) + High usage (>30 uses)
- **💼 Business User**: Moderate spend (>$2000) + Feature adoption (>15 features)
- **👑 Loyal User**: Long-term customers (>365 days)
- **👤 Standard User**: Regular usage patterns

## 🚀 Deployment

### **Streamlit Cloud (Recommended)**
1. Push repository to GitHub
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy with automatic updates
4. Access via custom URL

### **Local Development**
```bash
# Clone repository
git clone https://github.com/your-username/subsify.git
cd subsify

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run streamlit_app.py
```

### **Docker Deployment**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501"]
```

## 🛠️ Technical Architecture

### **Dependencies**
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning models and preprocessing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **Joblib**: Model serialization

### **Performance Optimization**
- **Caching**: `@st.cache_resource` for model loading
- **Session State**: Persistent user data and history
- **Lazy Loading**: Models loaded only when needed
- **Efficient Processing**: Vectorized operations for batch predictions

## 📁 Project Structure

```
subsify/
├── streamlit_app.py                          # Main Streamlit application
├── naive_bayes_model.py                     # Naive Bayes training script
├── neural_network_model.py                 # Neural Network training script
├── naive_bayes_subscription_model.joblib    # Trained Naive Bayes model
├── naive_bayes_scaler.joblib                # Feature scaler for Naive Bayes
├── neural_network_subscription_model.joblib # Trained Neural Network model
├── neural_network_scaler.joblib             # Feature scaler for Neural Network
├── requirements.txt                         # Python dependencies
├── .python-version                          # Python version specification
├── .gitignore                              # Git ignore patterns
└── README.md                               # Project documentation
```

## 🔧 Configuration

### **Environment Variables**
```bash
# Optional: Set custom configurations
export STREAMLIT_THEME_BASE="light"
export STREAMLIT_THEME_PRIMARY_COLOR="#FF6B6B"
export STREAMLIT_SERVER_PORT=8501
```

### **Model Retraining**
```bash
# Retrain Naive Bayes model
python naive_bayes_model.py

# Retrain Neural Network model
python neural_network_model.py
```

## 🎯 Business Value

### **For Data Scientists**
- **Model Comparison**: A/B testing framework for algorithm evaluation
- **Feature Analysis**: Built-in feature importance and correlation analysis
- **Performance Monitoring**: Real-time model usage and accuracy tracking
- **Experimentation**: Easy model switching and ensemble methods

### **For Business Users**
- **Customer Insights**: Actionable recommendations for each customer segment
- **Batch Processing**: Scalable predictions for large customer bases
- **Risk Assessment**: Confidence scores for prediction reliability
- **Strategic Planning**: Historical trends and pattern analysis

### **For Product Teams**
- **User Segmentation**: Automatic classification of customer types
- **Churn Prevention**: Early identification of at-risk customers
- **Revenue Optimization**: Focus on high-value prospects
- **Feature Usage**: Understanding product adoption patterns

## 📈 Future Enhancements

- **Real-time Data Integration**: Connect to live customer databases
- **Advanced Visualizations**: Interactive dashboards with drill-down capabilities
- **Model Monitoring**: Automated model drift detection and alerts
- **API Endpoints**: RESTful API for programmatic access
- **Custom Models**: User-uploadable model training capabilities
- **Advanced Segmentation**: ML-powered customer clustering

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the web interface
- Machine learning powered by [Scikit-learn](https://scikit-learn.org/)
- Visualizations created with [Plotly](https://plotly.com/)
- Deployed on [Streamlit Cloud](https://streamlit.io/cloud)

---

**Made with ❤️ by the Subsify Team**
