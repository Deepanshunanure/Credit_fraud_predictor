import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Financial Fraud Detection", page_icon="🕵️", layout="centered")

st.title("🕵️ Financial Fraud Detection")
st.write("This app uses your trained machine learning model to predict whether a transaction is **fraudulent** or **genuine**.")

MODEL_PATH = "best_fraud_detection_model.pkl"

# 1. Load the trained pipeline
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}. Make sure it's in the same folder as app.py.")
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

st.markdown("---")

# 2. Input form for transaction features
st.subheader("Enter Transaction Details")

with st.form("fraud_form"):
    col1, col2 = st.columns(2)

    with col1:
        step = st.number_input("Step (time step since start)", min_value=0, value=1)
        amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
        oldbalanceOrg = st.number_input("Sender Old Balance (oldbalanceOrg)", min_value=0.0, value=0.0)
        newbalanceOrg = st.number_input("Sender New Balance (newbalanceOrig)", min_value=0.0, value=0.0)
        oldbalanceDest = st.number_input("Receiver Old Balance (oldbalanceDest)", min_value=0.0, value=0.0)
        newbalanceDest = st.number_input("Receiver New Balance (newbalanceDest)", min_value=0.0, value=0.0)
        unusuallogin = st.number_input("Unusual Login Count (unusuallogin)", min_value=0, value=0)
        isFlaggedFraud = st.selectbox("Flagged by Rule Engine? (isFlaggedFraud)", options=[0, 1], index=0)

    with col2:
        tx_type = st.selectbox(
            "Transaction Type (type)",
            ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN", "OTHER"]
        )
        branch = st.text_input("Branch (branch)", value="Main")
        acct_type = st.selectbox(
            "Account Type (Acct type)",
            ["Savings", "Current", "Salary", "Credit", "Other"]
        )
        time_of_day = st.selectbox(
            "Time of Day (Time of day)",
            ["Morning", "Afternoon", "Evening", "Night"]
        )
        day_of_week = st.selectbox(
            "DayOfWeek(new)",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )

    submitted = st.form_submit_button("🔍 Predict Fraud")

if submitted:
    # 3. Compute engineered features EXACTLY as in training:
    # df["balanceDiffOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    # df["balanceDiffDest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    balanceDiffOrig = oldbalanceOrg - newbalanceOrg
    balanceDiffDest = newbalanceDest - oldbalanceDest

    # 4. Build a DataFrame with EXACT column names as used for training
    input_data = pd.DataFrame([{
        "step": step,
        "type": tx_type,
        "branch": branch,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrg,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "unusuallogin": unusuallogin,
        "isFlaggedFraud": isFlaggedFraud,
        "Acct type": acct_type,
        "Time of day": time_of_day,
        "DayOfWeek(new)": day_of_week,
        "balanceDiffOrig": balanceDiffOrig,
        "balanceDiffDest": balanceDiffDest,
    }])

    # 5. Predict
    try:
        pred = model.predict(input_data)[0]                # 0 = genuine, 1 = fraud
        proba = model.predict_proba(input_data)[0][1]      # probability of fraud
    except Exception as e:
        st.error(f"Error during prediction: {e}")
    else:
        st.markdown("---")
        st.subheader("Prediction Result")

        if pred == 1:
            st.error("⚠️ This transaction is predicted as **FRAUDULENT**.")
        else:
            st.success("✅ This transaction is predicted as **GENUINE**.")

        st.metric("Fraud Probability", f"{proba*100:.2f}%")

        with st.expander("See input data used for prediction"):
            st.write(input_data)
