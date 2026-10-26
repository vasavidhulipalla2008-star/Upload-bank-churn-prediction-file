import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Bank Customer Churn Prediction")
st.markdown(
    "A machine-learning mini project comparing four algorithms and predicting "
    "whether a bank customer may leave."
)

@st.cache_data
def load_data():
    return pd.read_csv("BankChurners.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error("BankChurners.csv was not found. Keep it in the same folder as app.py.")
    st.stop()

if not {"Attrition_Flag", "CLIENTNUM"}.issubset(df.columns):
    st.error("The CSV does not contain the expected BankChurners columns.")
    st.stop()

X = df.drop(columns=["Attrition_Flag", "CLIENTNUM"])
y = df["Attrition_Flag"]
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

@st.cache_resource
def train_models(_X_train, _y_train):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols)
    ])

    definitions = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42,
            class_weight="balanced", n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, random_state=42,
            class_weight="balanced", n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1,
            max_depth=3, random_state=42
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.1,
            max_leaf_nodes=31, random_state=42
        ),
    }

    fitted = {}
    for name, classifier in definitions.items():
        fitted[name] = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])
        fitted[name].fit(_X_train, _y_train)
    return fitted

with st.spinner("Training four machine-learning models..."):
    models = train_models(X_train, y_train)

predictions = {
    name: model.predict(X_test)
    for name, model in models.items()
}

rows = []
for name, pred in predictions.items():
    rows.append({
        "Algorithm": name,
        "Accuracy (%)": 100 * accuracy_score(y_test, pred),
        "Precision - Attrited (%)": 100 * precision_score(
            y_test, pred, pos_label="Attrited Customer", zero_division=0
        ),
        "Recall - Attrited (%)": 100 * recall_score(
            y_test, pred, pos_label="Attrited Customer", zero_division=0
        ),
        "F1-score - Attrited (%)": 100 * f1_score(
            y_test, pred, pos_label="Attrited Customer", zero_division=0
        ),
    })

metrics_df = pd.DataFrame(rows).sort_values(
    "Accuracy (%)", ascending=False
).reset_index(drop=True)
best_name = metrics_df.iloc[0]["Algorithm"]
best_accuracy = metrics_df.iloc[0]["Accuracy (%)"]

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Predict a Customer", "About the Project"]
)

if page == "Dashboard":
    st.header("📊 Model Performance Dashboard")
    total = len(df)
    attrited = int((y == "Attrited Customer").sum())
    existing = int((y == "Existing Customer").sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total customers", f"{total:,}")
    c2.metric("Attrited customers", f"{attrited:,}")
    c3.metric("Best test accuracy", f"{best_accuracy:.2f}%")

    left, right = st.columns(2)

    with left:
        st.subheader("Customer churn distribution")
        counts = y.value_counts().reindex(
            ["Existing Customer", "Attrited Customer"], fill_value=0
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(counts.index, counts.values)
        ax.set_ylabel("Number of customers")
        ax.set_title("Customer Status")
        ax.tick_params(axis="x", rotation=12)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Model accuracy comparison")
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(metrics_df["Algorithm"], metrics_df["Accuracy (%)"])
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Test Accuracy by Algorithm")
        ax.tick_params(axis="x", rotation=25)
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.5,
                f"{height:.2f}%",
                ha="center",
                fontsize=8
            )
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Performance comparison")
    st.dataframe(metrics_df.round(2), use_container_width=True, hide_index=True)
    st.success(
        f"Best model by test accuracy: {best_name} ({best_accuracy:.2f}%)."
    )
    st.caption(
        "Accuracy alone may not be enough. Check recall for attrited customers "
        "before deciding which model is most useful."
    )

    st.subheader("Confusion matrix")
    chosen = st.selectbox("Choose a model", list(models.keys()))
    labels = ["Attrited Customer", "Existing Customer"]
    cm = confusion_matrix(y_test, predictions[chosen], labels=labels)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    image = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="Actual label",
        title=f"Confusion Matrix - {chosen}"
    )
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black"
            )
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Classification report")
    st.code(classification_report(
        y_test, predictions[chosen], zero_division=0
    ))

elif page == "Predict a Customer":
    st.header("🧑‍💼 Predict Customer Churn")
    st.write("Enter the customer's information and select an algorithm.")
    st.info("This is a model estimate, not a guarantee of future behaviour.")

    with st.form("customer_form"):
        values = {}
        col1, col2 = st.columns(2)

        for i, column in enumerate(X.columns):
            target_col = col1 if i % 2 == 0 else col2
            with target_col:
                if column in numeric_cols:
                    series = pd.to_numeric(X[column], errors="coerce").dropna()
                    minimum = float(series.min())
                    maximum = float(series.max())
                    default = float(series.median())
                    values[column] = st.number_input(
                        column.replace("_", " "),
                        min_value=minimum,
                        max_value=maximum,
                        value=min(max(default, minimum), maximum),
                        help=f"Dataset range: {minimum:g} to {maximum:g}"
                    )
                else:
                    options = sorted(X[column].dropna().astype(str).unique().tolist())
                    values[column] = st.selectbox(
                        column.replace("_", " "), options
                    )

        model_choice = st.selectbox(
            "Prediction algorithm",
            list(models.keys()),
            index=list(models.keys()).index(best_name)
        )
        submitted = st.form_submit_button(
            "Predict Customer Churn", type="primary"
        )

    if submitted:
        customer = pd.DataFrame([values], columns=X.columns)
        selected_model = models[model_choice]
        prediction = selected_model.predict(customer)[0]

        if prediction == "Attrited Customer":
            st.warning(
                "Prediction: Attrited Customer — the model classifies "
                "this customer as likely to leave."
            )
        else:
            st.success(
                "Prediction: Existing Customer — the model classifies "
                "this customer as likely to remain."
            )

        if hasattr(selected_model, "predict_proba"):
            probabilities = selected_model.predict_proba(customer)[0]
            classes = selected_model.named_steps["classifier"].classes_
            prob_df = pd.DataFrame({
                "Customer status": classes,
                "Model probability (%)": probabilities * 100
            }).sort_values("Model probability (%)", ascending=False)

            st.subheader("Model probability estimates")
            st.dataframe(prob_df.round(2), use_container_width=True, hide_index=True)
            st.caption("Probabilities are model estimates, not guarantees.")

elif page == "About the Project":
    st.header("ℹ️ About this mini project")
    st.markdown("""
**Objective:** Predict whether a bank customer is classified as attrited or existing.

**Dataset:** `BankChurners.csv`

**Algorithms compared:**
1. Random Forest
2. Extra Trees
3. Gradient Boosting
4. HistGradientBoosting

**Evaluation:** Test accuracy, precision, recall, F1-score, and confusion matrix.

**Visualization:** Matplotlib only.

**Tools:** Python, Pandas, Scikit-learn, Matplotlib, and Streamlit.
""")
    st.warning(
        "This is an educational demonstration, not a production banking decision "
        "system. Do not publish confidential customer data."
    )

st.divider()
st.caption("B.Tech Mini Project • Bank Customer Churn Prediction • Matplotlib + Streamlit")
