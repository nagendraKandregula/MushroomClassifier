import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Mushroom Classification",
    page_icon="🍄",
    layout="wide"
)

st.title("🍄 Mushroom Classification Dashboard")
st.write(
    "Evaluate mushroom classification models using "
    "your uploaded test dataset."
)

# --------------------------------------------------
# 2. Model configuration
# --------------------------------------------------

MODEL_DIR = Path("models")

MODEL_PATHS = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "K-Nearest Neighbors": MODEL_DIR / "knn.joblib",
    "Naive Bayes": MODEL_DIR / "naive_bayes.joblib",
    "Support Vector Machine": MODEL_DIR / "svm.joblib"
}

FEATURES = [
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat"
]

# --------------------------------------------------
# 3. Load selected model
# --------------------------------------------------

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


# --------------------------------------------------
# 4. Sidebar - model selection and file upload
# --------------------------------------------------

st.sidebar.header("Model Configuration")

selected_model_name = st.sidebar.selectbox(
    "Select Classification Model",
    list(MODEL_PATHS.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV) with no headers",
    type=["csv"]
)

# --------------------------------------------------
# 5. Main evaluation workflow
# --------------------------------------------------

if uploaded_file is not None:

    try:
        columns = [
            "class", "cap-shape", "cap-surface", "cap-color", "bruises",
            "odor", "gill-attachment", "gill-spacing", "gill-size",
            "gill-color", "stalk-shape", "stalk-root",
            "stalk-surface-above-ring", "stalk-surface-below-ring",
            "stalk-color-above-ring", "stalk-color-below-ring",
            "veil-type", "veil-color", "ring-number", "ring-type",
            "spore-print-color", "population", "habitat"
        ]

        data = pd.read_csv(uploaded_file, header=None, names=columns)

        st.subheader("Uploaded Dataset")
        st.dataframe(data.head(), use_container_width=True)

        # Validate dataset columns
        required_columns = FEATURES + ["class"]

        if missing_columns := [
            col for col in required_columns
            if col not in data.columns
        ]:
            st.error(
                f"Missing required columns: {missing_columns}"
            )
            st.stop()

        # Separate input features and actual labels
        X_test = data[FEATURES]
        y_test = data["class"]

        # Validate target labels
        if not set(y_test.dropna().unique()).issubset({"e", "p"}):
            st.error("Target column must contain only 'e' and 'p'.")
            st.stop()

        if y_test.isna().any():
            st.error("Target column contains missing values.")
            st.stop()

        # Load selected model
        model_path = MODEL_PATHS[selected_model_name]

        if not model_path.exists():
            st.error(
                f"Model file not found: {model_path}"
            )
            st.stop()

        model = load_model(str(model_path))

        # Generate predictions
        y_pred = model.predict(X_test)

        # Get probability of poisonous class
        classes = list(model.classes_)
        poisonous_index = classes.index("p")
        y_prob = model.predict_proba(X_test)[:, poisonous_index]

        # --------------------------------------------------
        # 6. Calculate evaluation metrics
        # --------------------------------------------------

        accuracy = accuracy_score(y_test, y_pred)

        auc = roc_auc_score(
            (y_test == "p").astype(int),
            y_prob
        )

        precision = precision_score(
            y_test, y_pred,
            pos_label="p",
            zero_division=0
        )

        recall = recall_score(
            y_test, y_pred,
            pos_label="p",
            zero_division=0
        )

        f1 = f1_score(
            y_test, y_pred,
            pos_label="p",
            zero_division=0
        )

        mcc = matthews_corrcoef(y_test, y_pred)

        # --------------------------------------------------
        # 7. Display metrics
        # --------------------------------------------------

        st.subheader(f"Evaluation Results: {selected_model_name}")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("AUC", f"{auc:.4f}")
        col3.metric("Precision", f"{precision:.4f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("Recall", f"{recall:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")

        # --------------------------------------------------
        # 8. Confusion matrix
        # --------------------------------------------------

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=["e", "p"]
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Edible", "Poisonous"],
            yticklabels=["Edible", "Poisonous"],
            ax=ax
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")

        st.pyplot(fig)
        plt.close(fig)

        # --------------------------------------------------
        # 9. Classification report
        # --------------------------------------------------

        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            labels=["e", "p"],
            target_names=["Edible", "Poisonous"],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )

        # --------------------------------------------------
        # 10. Download predictions
        # --------------------------------------------------

        results = X_test.copy()
        results["Actual Class"] = y_test
        results["Predicted Class"] = y_pred

        results["Predicted Probability (Poisonous)"] = y_prob

        csv = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Prediction Results",
            data=csv,
            file_name="mushroom_predictions.csv",
            mime="text/csv"
        )

    except Exception as error:
        st.error(f"An error occurred: {error}")

else:
    st.info(
        "Please upload a test CSV file using the sidebar "
        "to begin model evaluation."
    )
