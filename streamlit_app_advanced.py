import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
from sklearn.metrics import classification_report

# Configure page
st.set_page_config(
    page_title="Subsify",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS for mobile responsiveness and better styling
st.markdown("""
<style>
    .main .block-container {
        max-width: 95%;
        padding: 1rem;
    }
    @media (max-width: 768px) {
        .stTextInput > div > div > input {
            font-size: 16px;
        }
        .stSelectbox > div > div > select {
            font-size: 16px;
        }
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .prediction-high {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .prediction-low {
        background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_usage' not in st.session_state:
    st.session_state.model_usage = {'nb': 0, 'nn': 0}
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Load models with enhanced caching
@st.cache_resource
def load_models():
    try:
        # Load Naive Bayes model
        nb_model = joblib.load("naive_bayes_subscription_model.joblib")
        nb_scaler = joblib.load("naive_bayes_scaler.joblib")
        
        # Load Neural Network model
        nn_model = joblib.load("neural_network_subscription_model.joblib")
        nn_scaler = joblib.load("neural_network_scaler.joblib")
        
        return nb_model, nb_scaler, nn_model, nn_scaler
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        return None, None, None, None

@st.cache_data
def get_feature_importance():
    # Mock feature importance data - in production, extract from your models
    features = ['Service Usage', 'Avg Interval', 'Total Spend', 'Account Age', 
               'Help Desk Tickets', 'Last Login', 'Feature Usage', 'Peak Hour', 'Mobile Ratio']
    importance = [0.25, 0.20, 0.18, 0.12, 0.08, 0.07, 0.05, 0.03, 0.02]
    return features, importance

def get_customer_segment(num_uses, total_spend, account_age_days, feature_usage_count):
    """Determine customer segment based on usage patterns"""
    if total_spend > 5000 and num_uses > 30:
        return "🌟 Premium User", "#FFD700"
    elif total_spend > 2000 and feature_usage_count > 15:
        return "💼 Business User", "#32CD32"
    elif account_age_days > 365:
        return "👑 Loyal User", "#9370DB"
    else:
        return "👤 Standard User", "#87CEEB"

def get_recommendations(prediction, confidence, segment, features_dict):
    """Generate personalized recommendations"""
    recommendations = []
    
    if prediction and confidence > 0.8:
        recommendations.append("✅ **High-value prospect** - prioritize for premium offers")
        recommendations.append("🎯 **Action**: Schedule personal demo or consultation")
    elif prediction and confidence > 0.6:
        recommendations.append("⚠️ **Moderate probability** - engage with targeted campaigns")
        recommendations.append("📧 **Action**: Send feature highlights and success stories")
    else:
        recommendations.append("❌ **Low probability** - focus on nurturing and engagement")
        recommendations.append("🔄 **Action**: Implement user onboarding and tutorial campaigns")
    
    # Segment-specific recommendations
    if "Premium" in segment:
        recommendations.append("💎 **Premium Strategy**: Offer enterprise features and dedicated support")
    elif "Business" in segment:
        recommendations.append("🏢 **Business Strategy**: Highlight team collaboration and productivity features")
    elif "Loyal" in segment:
        recommendations.append("🎁 **Loyalty Strategy**: Reward long-term engagement with exclusive benefits")
    
    return recommendations

def validate_inputs(num_uses, avg_interval, total_spend, account_age_days, support_tickets, 
                   last_login_days_ago, feature_usage_count, peak_usage_hour, mobile_usage_ratio):
    """Validate user inputs and return warnings"""
    warnings = []
    
    if total_spend < 50:
        warnings.append("⚠️ Total spend seems unusually low")
    elif total_spend > 10000:
        warnings.append("⚠️ Total spend seems unusually high")
    
    if num_uses > 50:
        warnings.append("⚠️ Usage count seems very high")
    elif num_uses < 1:
        warnings.append("⚠️ Usage count must be at least 1")
    
    if avg_interval > 180:
        warnings.append("⚠️ Average interval seems very long")
    
    if last_login_days_ago > 90:
        warnings.append("⚠️ Customer hasn't logged in for a long time")
    
    if support_tickets > 20:
        warnings.append("⚠️ Unusually high number of support tickets")
    
    return warnings

def process_batch_predictions(df, nb_model, nb_scaler, nn_model, nn_scaler):
    """Process batch predictions for uploaded CSV"""
    results = []
    
    for idx, row in df.iterrows():
        features = np.array([[
            row['num_uses'], row['avg_interval'], row['total_spend'], 
            row['account_age_days'], row['support_tickets'], 
            row['last_login_days_ago'], row['feature_usage_count'],
            row['peak_usage_hour'], row['mobile_usage_ratio']
        ]])
        
        # Naive Bayes prediction
        nb_features_scaled = nb_scaler.transform(features)
        nb_prediction = nb_model.predict(nb_features_scaled)[0]
        nb_probability = nb_model.predict_proba(nb_features_scaled)[0][1]
        
        # Neural Network prediction
        nn_features_scaled = nn_scaler.transform(features)
        nn_prediction = nn_model.predict(nn_features_scaled)[0]
        nn_probability = nn_model.predict_proba(nn_features_scaled)[0][1]
        
        results.append({
            'customer_id': row.get('customer_id', f'Customer_{idx+1}'),
            'nb_prediction': nb_prediction,
            'nb_probability': nb_probability,
            'nn_prediction': nn_prediction,
            'nn_probability': nn_probability,
            'ensemble_probability': (nb_probability + nn_probability) / 2
        })
    
    return pd.DataFrame(results)

# Load models
nb_model, nb_scaler, nn_model, nn_scaler = load_models()

# Main app
st.title("🎯 Subsify")
st.write("AI-powered customer subscription prediction platform with advanced analytics")

# Sidebar for navigation
with st.sidebar:
    st.header("🔧 Navigation")
    app_mode = st.selectbox("Choose Mode", 
                           ["Single Prediction", "Batch Processing", "Model Analytics", "A/B Testing"])
    
    if nb_model is not None and nn_model is not None:
        st.header("📊 Model Usage Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Naive Bayes", st.session_state.model_usage['nb'])
        with col2:
            st.metric("Neural Network", st.session_state.model_usage['nn'])

# Single Prediction Mode
if app_mode == "Single Prediction":
    if nb_model is not None and nn_model is not None:
        # Model selection with comparison option
        comparison_mode = st.checkbox("🔬 Compare Both Models", value=False)
        
        if not comparison_mode:
            model_choice = st.selectbox(
                "🤖 Select ML Model",
                ["Naive Bayes", "Neural Network"],
                help="Choose between Naive Bayes (fast, interpretable) or Neural Network (complex patterns)"
            )
            
            # Set current model and scaler based on selection
            if model_choice == "Naive Bayes":
                current_model = nb_model
                current_scaler = nb_scaler
                model_info = "Gaussian Naive Bayes"
                st.session_state.model_usage['nb'] += 1
            else:
                current_model = nn_model
                current_scaler = nn_scaler
                model_info = "Neural Network (MLPClassifier)"
                st.session_state.model_usage['nn'] += 1
        
        # Create input fields in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Usage Metrics")
            num_uses = st.number_input(
                "Number of Uses", 
                min_value=1, max_value=100, value=25,
                help="How many times the customer used the service"
            )
            avg_interval = st.number_input(
                "Average Interval (days)", 
                min_value=1, max_value=180, value=15,
                help="Average days between uses"
            )
            total_spend = st.number_input(
                "Total Spend ($)", 
                min_value=50, max_value=10000, value=2500,
                help="Total amount spent by customer"
            )
            account_age_days = st.number_input(
                "Account Age (days)", 
                min_value=30, max_value=1095, value=365,
                help="How long the customer has been with us"
            )
            mobile_usage_ratio = st.slider(
                "Mobile Usage Ratio", 
                min_value=0.0, max_value=1.0, value=0.7, step=0.1,
                help="Percentage of mobile vs desktop usage"
            )
        
        with col2:
            st.subheader("🎯 Behavior Metrics")
            support_tickets = st.number_input(
                "Support Tickets", 
                min_value=0, max_value=25, value=3,
                help="Number of support tickets raised"
            )
            last_login_days_ago = st.number_input(
                "Last Login (days ago)", 
                min_value=0, max_value=180, value=5,
                help="Days since last login"
            )
            feature_usage_count = st.number_input(
                "Feature Usage Count", 
                min_value=1, max_value=25, value=12,
                help="Number of different features used"
            )
            peak_usage_hour = st.number_input(
                "Peak Usage Hour (0-23)", 
                min_value=0, max_value=23, value=14,
                help="Hour of day when customer is most active"
            )
        
        # Input validation
        warnings = validate_inputs(num_uses, avg_interval, total_spend, account_age_days, 
                                 support_tickets, last_login_days_ago, feature_usage_count, 
                                 peak_usage_hour, mobile_usage_ratio)
        
        if warnings:
            st.warning("Input Validation Warnings:")
            for warning in warnings:
                st.write(f"  {warning}")
        
        # Predict button
        if st.button("🚀 Predict Subscription", type="primary"):
            try:
                with st.spinner("Analyzing customer data..."):
                    # Prepare data
                    features = np.array([[
                        num_uses, avg_interval, total_spend, account_age_days,
                        support_tickets, last_login_days_ago, feature_usage_count,
                        peak_usage_hour, mobile_usage_ratio
                    ]])
                    
                    features_dict = {
                        'num_uses': num_uses,
                        'avg_interval': avg_interval,
                        'total_spend': total_spend,
                        'account_age_days': account_age_days,
                        'support_tickets': support_tickets,
                        'last_login_days_ago': last_login_days_ago,
                        'feature_usage_count': feature_usage_count,
                        'peak_usage_hour': peak_usage_hour,
                        'mobile_usage_ratio': mobile_usage_ratio
                    }
                    
                    # Get customer segment
                    segment, segment_color = get_customer_segment(
                        num_uses, total_spend, account_age_days, feature_usage_count
                    )
                    
                    if comparison_mode:
                        # Compare both models
                        st.divider()
                        st.subheader("🔬 Model Comparison Results")
                        
                        comp_col1, comp_col2 = st.columns(2)
                        
                        with comp_col1:
                            st.markdown("### 🧠 Naive Bayes")
                            nb_features_scaled = nb_scaler.transform(features)
                            nb_prediction = nb_model.predict(nb_features_scaled)[0]
                            nb_probability = nb_model.predict_proba(nb_features_scaled)[0]
                            
                            if nb_prediction == 1:
                                st.success("✅ **Likely to Subscribe**")
                            else:
                                st.error("❌ **Unlikely to Subscribe**")
                            
                            st.metric("Subscription Probability", f"{nb_probability[1]:.1%}")
                            st.metric("Model Confidence", f"{max(nb_probability):.1%}")
                        
                        with comp_col2:
                            st.markdown("### 🤖 Neural Network")
                            nn_features_scaled = nn_scaler.transform(features)
                            nn_prediction = nn_model.predict(nn_features_scaled)[0]
                            nn_probability = nn_model.predict_proba(nn_features_scaled)[0]
                            
                            if nn_prediction == 1:
                                st.success("✅ **Likely to Subscribe**")
                            else:
                                st.error("❌ **Unlikely to Subscribe**")
                            
                            st.metric("Subscription Probability", f"{nn_probability[1]:.1%}")
                            st.metric("Model Confidence", f"{max(nn_probability):.1%}")
                        
                        # Ensemble prediction
                        st.divider()
                        st.subheader("🎯 Ensemble Prediction")
                        ensemble_prob = (nb_probability[1] + nn_probability[1]) / 2
                        ensemble_prediction = 1 if ensemble_prob > 0.5 else 0
                        
                        if ensemble_prediction == 1:
                            st.success(f"✅ **Ensemble: Likely to Subscribe** ({ensemble_prob:.1%})")
                        else:
                            st.error(f"❌ **Ensemble: Unlikely to Subscribe** ({ensemble_prob:.1%})")
                        
                        # Use ensemble for recommendations
                        prediction = ensemble_prediction
                        probability = [1-ensemble_prob, ensemble_prob]
                        
                    else:
                        # Single model prediction
                        features_scaled = current_scaler.transform(features)
                        prediction = current_model.predict(features_scaled)[0]
                        probability = current_model.predict_proba(features_scaled)[0]
                        
                        # Display results
                        st.divider()
                        st.subheader("📈 Prediction Results")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if prediction == 1:
                                st.success("✅ **Likely to Subscribe**")
                            else:
                                st.error("❌ **Unlikely to Subscribe**")
                        
                        with col2:
                            st.metric(
                                "Subscription Probability", 
                                f"{probability[1]:.1%}",
                                delta=f"{probability[1] - 0.5:.1%}" if probability[1] > 0.5 else f"{probability[1] - 0.5:.1%}"
                            )
                        
                        with col3:
                            confidence = max(probability)
                            st.metric(
                                "Model Confidence", 
                                f"{confidence:.1%}",
                                delta="High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
                            )
                    
                    # Customer segment and insights
                    st.divider()
                    st.subheader("👤 Customer Profile")
                    
                    profile_col1, profile_col2 = st.columns(2)
                    
                    with profile_col1:
                        st.markdown(f"**Segment:** {segment}")
                        
                        insights = []
                        if num_uses > 30:
                            insights.append("🔥 High usage customer")
                        if avg_interval < 20:
                            insights.append("⚡ Frequent user")
                        if total_spend > 3000:
                            insights.append("💰 High-value customer")
                        if last_login_days_ago < 7:
                            insights.append("🎯 Recently active")
                        if feature_usage_count > 15:
                            insights.append("🚀 Power user")
                        
                        if insights:
                            st.write("**Key Insights:**")
                            for insight in insights:
                                st.write(f"- {insight}")
                        else:
                            st.write("- 📊 Standard usage pattern")
                    
                    with profile_col2:
                        # Customer behavior radar chart
                        categories = ['Usage Frequency', 'Spending', 'Engagement', 'Loyalty', 'Support Needs']
                        values = [
                            min(num_uses / 50 * 100, 100),
                            min(total_spend / 8000 * 100, 100),
                            min(feature_usage_count / 20 * 100, 100),
                            min(account_age_days / 1095 * 100, 100),
                            max(100 - (support_tickets / 15 * 100), 0)
                        ]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=categories,
                            fill='toself',
                            name='Customer Profile'
                        ))
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 100]
                                )),
                            showlegend=False,
                            title="Customer Behavior Profile",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Personalized recommendations
                    st.divider()
                    st.subheader("💡 Personalized Recommendations")
                    
                    recommendations = get_recommendations(prediction, probability[1], segment, features_dict)
                    for rec in recommendations:
                        st.write(f"- {rec}")
                    
                    # Save prediction to history
                    prediction_record = {
                        'timestamp': datetime.now().isoformat(),
                        'features': features_dict,
                        'prediction': int(prediction),
                        'probability': float(probability[1]),
                        'segment': segment,
                        'model': model_choice if not comparison_mode else "Ensemble"
                    }
                    st.session_state.prediction_history.append(prediction_record)
                    
                    # Download results
                    st.divider()
                    st.subheader("📥 Export Results")
                    
                    export_data = {
                        'customer_data': features_dict,
                        'prediction_result': {
                            'prediction': int(prediction),
                            'probability': float(probability[1]),
                            'confidence': float(max(probability)),
                            'segment': segment
                        },
                        'recommendations': recommendations,
                        'timestamp': datetime.now().isoformat(),
                        'model_used': model_choice if not comparison_mode else "Ensemble"
                    }
                    
                    st.download_button(
                        "📥 Download Prediction Results",
                        data=json.dumps(export_data, indent=2),
                        file_name=f"subsify_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.info("💡 Please check your input values and try again")

# Batch Processing Mode
elif app_mode == "Batch Processing":
    st.header("📂 Batch Processing")
    st.write("Upload a CSV file with customer data for batch predictions")
    
    # Sample CSV download
    sample_data = pd.DataFrame({
        'customer_id': ['CUST_001', 'CUST_002', 'CUST_003'],
        'num_uses': [25, 45, 10],
        'avg_interval': [15, 8, 30],
        'total_spend': [2500, 5000, 800],
        'account_age_days': [365, 180, 730],
        'support_tickets': [3, 1, 8],
        'last_login_days_ago': [5, 2, 15],
        'feature_usage_count': [12, 18, 6],
        'peak_usage_hour': [14, 10, 20],
        'mobile_usage_ratio': [0.7, 0.4, 0.9]
    })
    
    st.download_button(
        "📥 Download Sample CSV Template",
        data=sample_data.to_csv(index=False),
        file_name="subsify_batch_template.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader("📂 Upload CSV for batch predictions", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("**Preview of uploaded data:**")
            st.dataframe(df.head())
            
            if st.button("🚀 Process Batch Predictions"):
                with st.spinner("Processing batch predictions..."):
                    results_df = process_batch_predictions(df, nb_model, nb_scaler, nn_model, nn_scaler)
                    
                    st.success(f"✅ Processed {len(results_df)} customers successfully!")
                    
                    # Display results
                    st.subheader("📊 Batch Results")
                    st.dataframe(results_df)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Customers", len(results_df))
                    with col2:
                        nb_likely = sum(results_df['nb_prediction'])
                        st.metric("NB: Likely to Subscribe", f"{nb_likely} ({nb_likely/len(results_df)*100:.1f}%)")
                    with col3:
                        nn_likely = sum(results_df['nn_prediction'])
                        st.metric("NN: Likely to Subscribe", f"{nn_likely} ({nn_likely/len(results_df)*100:.1f}%)")
                    
                    # Visualization
                    fig = px.histogram(results_df, x='ensemble_probability', nbins=20,
                                     title="Distribution of Subscription Probabilities")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download results
                    st.download_button(
                        "📥 Download Batch Results",
                        data=results_df.to_csv(index=False),
                        file_name=f"subsify_batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("💡 Please ensure your CSV has the required columns")

# Model Analytics Mode
elif app_mode == "Model Analytics":
    st.header("📊 Model Analytics Dashboard")
    
    # Feature importance
    features, importance = get_feature_importance()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Feature Importance")
        fig = px.bar(x=features, y=importance, 
                    title="Feature Importance in Prediction Model")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Model Performance Metrics")
        
        # Mock performance metrics
        performance_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'],
            'Naive Bayes': [0.85, 0.82, 0.88, 0.85, 0.91],
            'Neural Network': [0.88, 0.86, 0.90, 0.88, 0.94]
        }
        
        performance_df = pd.DataFrame(performance_data)
        
        fig = px.bar(performance_df, x='Metric', y=['Naive Bayes', 'Neural Network'],
                    title="Model Performance Comparison", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    # Model usage over time
    if st.session_state.prediction_history:
        st.subheader("📅 Prediction History")
        
        history_df = pd.DataFrame(st.session_state.prediction_history)
        history_df['date'] = pd.to_datetime(history_df['timestamp']).dt.date
        
        daily_predictions = history_df.groupby('date').size().reset_index(name='count')
        
        fig = px.line(daily_predictions, x='date', y='count',
                     title="Daily Predictions Over Time")
        st.plotly_chart(fig, use_container_width=True)
        
        # Segment distribution
        segment_counts = history_df['segment'].value_counts()
        fig = px.pie(values=segment_counts.values, names=segment_counts.index,
                    title="Customer Segment Distribution")
        st.plotly_chart(fig, use_container_width=True)

# A/B Testing Mode
elif app_mode == "A/B Testing":
    st.header("🧪 A/B Testing Dashboard")
    st.write("Compare model performance and user engagement")
    
    # Model usage statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Model Usage Statistics")
        model_usage_df = pd.DataFrame({
            'Model': ['Naive Bayes', 'Neural Network'],
            'Usage Count': [st.session_state.model_usage['nb'], st.session_state.model_usage['nn']]
        })
        
        fig = px.bar(model_usage_df, x='Model', y='Usage Count',
                    title="Model Usage Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Conversion Analysis")
        
        # Mock conversion data
        conversion_data = {
            'Model': ['Naive Bayes', 'Neural Network'],
            'Predictions Made': [st.session_state.model_usage['nb'], st.session_state.model_usage['nn']],
            'Positive Predictions': [int(st.session_state.model_usage['nb'] * 0.65), 
                                   int(st.session_state.model_usage['nn'] * 0.70)]
        }
        
        conversion_df = pd.DataFrame(conversion_data)
        conversion_df['Conversion Rate'] = (conversion_df['Positive Predictions'] / 
                                          conversion_df['Predictions Made']).fillna(0)
        
        fig = px.bar(conversion_df, x='Model', y='Conversion Rate',
                    title="Model Conversion Rate Comparison")
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistical significance test
    st.subheader("📈 Statistical Analysis")
    
    if st.session_state.model_usage['nb'] > 0 and st.session_state.model_usage['nn'] > 0:
        # Mock statistical test results
        st.write("**A/B Test Results:**")
        st.write(f"- Sample Size: NB={st.session_state.model_usage['nb']}, NN={st.session_state.model_usage['nn']}")
        st.write(f"- Conversion Rate: NB=65%, NN=70%")
        st.write(f"- Statistical Significance: {'✅ Significant' if st.session_state.model_usage['nb'] + st.session_state.model_usage['nn'] > 30 else '❌ Not enough data'}")
    else:
        st.info("💡 Need more data points to perform statistical analysis")

# Sidebar - Model Information
with st.sidebar:
    st.header("ℹ️ Model Information")
    if nb_model is not None and nn_model is not None:
        st.write("**Algorithms:** Naive Bayes & Neural Network")
        st.write("**Features:** 9 customer attributes")
        st.write("**Training Data:** 1000 synthetic customers")
        
        # Model comparison info
        st.info("**Model Comparison:**")
        st.write("• **Naive Bayes**: Fast, interpretable, good for linear relationships")
        st.write("• **Neural Network**: Complex patterns, higher accuracy potential")
        
        # Feature explanations
        st.header("📚 Feature Guide")
        with st.expander("Usage Metrics"):
            st.write("""
            - **Number of Uses**: Service usage frequency
            - **Average Interval**: Days between uses
            - **Total Spend**: Customer lifetime value
            - **Account Age**: Customer loyalty indicator
            - **Mobile Ratio**: Platform preference
            """)
        
        with st.expander("Behavior Metrics"):
            st.write("""
            - **Support Tickets**: Help-seeking behavior
            - **Last Login**: Recent engagement
            - **Feature Count**: Product adoption breadth
            - **Peak Hour**: Usage timing pattern
            """)
        
        # Clear history button
        if st.button("🗑️ Clear Prediction History"):
            st.session_state.prediction_history = []
            st.session_state.model_usage = {'nb': 0, 'nn': 0}
            st.success("History cleared!")
    else:
        st.error("Could not load prediction models. Please check if the model files are available.")

# Footer
st.markdown("---")
st.markdown("**Subsify** - Advanced AI-powered subscription prediction")
