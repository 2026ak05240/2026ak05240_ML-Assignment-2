import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

st.set_page_config(page_title="ML Classifier Evaluator", layout="wide")

st.title("📊 Multi-Model ML Classification Evaluator")
st.write("Evaluating pre-trained classification models on the test dataset (`data/sample_test_data.csv`).")

# Automatically load test data from the local directory path
@st.cache_data
def load_data():
    return pd.read_csv("data/sample_test_data.csv")

try:
    test_df = load_data()
    
    # Sidebar Model Selection Only
    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox(
        "Choose ML Model",
        ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors", "Naive Bayes", "Support Vector Machine"]
    )

    model_mapping = {
        "Logistic Regression": "logistic_regression",
        "Decision Tree": "decision_tree",
        "K-Nearest Neighbors": "knn",
        "Naive Bayes": "naive_bayes",
        "Support Vector Machine": "svm"
    }

    st.write("### Preview of Test Dataset", test_df.head())
    
    if 'target' not in test_df.columns:
        st.error("The loaded test CSV dataset must contain a 'target' column for evaluation.")
    else:
        X_test = test_df.drop(columns=['target'])
        y_test = test_df['target']
        
        # Load scaler and model
        scaler = joblib.load("models/scaler.pkl")
        model_key = model_mapping[model_choice]
        model = joblib.load(f"models/{model_key}.pkl")
        
        # Apply scaling for linear/distance models
        if model_key in ["logistic_regression", "knn", "svm"]:
            X_eval = scaler.transform(X_test)
        else:
            X_eval = X_test.values
        
        # Predictions
        y_pred = model.predict(X_eval)
        y_proba = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        # Metric Computations
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        st.markdown("---")
        st.subheader(f"Evaluation Results for **{model_choice}**")
        
        # Display metrics in columns
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{prec:.4f}")
        col4.metric("Recall", f"{rec:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")
        
        # Visualizations & Report
        st.markdown("---")
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            st.pyplot(fig)
            
        with col_right:
            st.subheader("Classification Report")
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report_dict).transpose().style.highlight_max(axis=0))

except Exception as e:
    st.error(f"Error loading test data or models: {e}. Please ensure you have run `train_and_save.py` first.")