import streamlit as st
import joblib
import numpy as np

# Configure page
st.set_page_config(
    page_title="Subscription Predictor",
    page_icon="🔮",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    try:
        nb_model = joblib.load("naive_bayes_subscription_model.joblib")
        scaler = joblib.load("naive_bayes_scaler.joblib")
        return nb_model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please make sure the .joblib files are in the same directory.")
        return None, None

nb_model, scaler = load_models()

# Main app
st.title("🔮 Subscription Predictor")
st.write("Enter customer data to predict subscription likelihood using Naive Bayes:")

if nb_model is not None and scaler is not None:
    # Create input fields in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Usage Metrics")
        num_uses = st.number_input(
            "Number of Uses", 
            min_value=1, max_value=50, value=25,
            help="How many times the customer used the service"
        )
        avg_interval = st.number_input(
            "Average Interval (days)", 
            min_value=1, max_value=120, value=15,
            help="Average days between uses"
        )
        total_spend = st.number_input(
            "Total Spend ($)", 
            min_value=50, max_value=8000, value=2500,
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
            min_value=0, max_value=15, value=3,
            help="Number of support tickets raised"
        )
        last_login_days_ago = st.number_input(
            "Last Login (days ago)", 
            min_value=0, max_value=90, value=5,
            help="Days since last login"
        )
        feature_usage_count = st.number_input(
            "Feature Usage Count", 
            min_value=1, max_value=20, value=12,
            help="Number of different features used"
        )
        peak_usage_hour = st.number_input(
            "Peak Usage Hour (0-23)", 
            min_value=0, max_value=23, value=14,
            help="Hour of day when customer is most active"
        )
    
    # Predict button
    if st.button("🚀 Predict Subscription", type="primary"):
        with st.spinner("Analyzing customer data..."):
            # Prepare data
            features = np.array([[
                num_uses, avg_interval, total_spend, account_age_days,
                support_tickets, last_login_days_ago, feature_usage_count,
                peak_usage_hour, mobile_usage_ratio
            ]])
            
            # Scale and predict
            features_scaled = scaler.transform(features)
            prediction = nb_model.predict(features_scaled)[0]
            probability = nb_model.predict_proba(features_scaled)[0]
            
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
            
            # Additional insights
            st.subheader("🔍 Customer Insights")
            
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
                for insight in insights:
                    st.write(f"- {insight}")
            else:
                st.write("- 📊 Standard usage pattern")
            
            # Recommendation
            st.subheader("💡 Recommendation")
            if prediction == 1:
                st.success("**Target for subscription offer!** This customer shows strong indicators for subscription conversion.")
            else:
                st.warning("**Nurture this customer** with engagement campaigns before offering subscription.")

    # Model information
    st.sidebar.header("ℹ️ Model Information")
    st.sidebar.write("**Algorithm:** Naive Bayes")
    st.sidebar.write("**Features:** 9 customer attributes")
    st.sidebar.write("**Training Data:** 1000 synthetic customers")
    
    # Feature explanations
    st.sidebar.header("📚 Feature Guide")
    with st.sidebar.expander("Usage Metrics"):
        st.write("""
        - **Number of Uses**: Service usage frequency
        - **Average Interval**: Days between uses
        - **Total Spend**: Customer lifetime value
        - **Account Age**: Customer loyalty indicator
        - **Mobile Ratio**: Platform preference
        """)
    
    with st.sidebar.expander("Behavior Metrics"):
        st.write("""
        - **Support Tickets**: Help-seeking behavior
        - **Last Login**: Recent engagement
        - **Feature Count**: Product adoption breadth
        - **Peak Hour**: Usage timing pattern
        """)

else:
    st.error("Could not load the prediction model. Please check if the model files are available.")
