import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# Define column names
columns = [
    "class",
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

# Import dataset
df = pd.read_csv(
    "D:\\BITS\\ML-Assignments\\ML-Assignment-2\\MushroomClassification\\mushroom.csv",
    header=None,
    names=columns
)

print(df.head())
print(df.shape)

# Dataset dimensions
print("Dataset shape:", df.shape)

# Column information
print(df.info())

# Statistical summary
print(df.describe(include="all"))

#checking missing values
print(df.isnull().sum())
print(
    df["stalk-root"].value_counts()
)

print(df["class"].value_counts())

print(
    df["class"].value_counts(normalize=True) * 100
)

sns.countplot(data=df, x="class")

plt.title("Mushroom Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Mushrooms")

#plt.show()

# Independent variables
X = df.drop("class", axis=1)

# Dependent variable
y = df["class"]

print("Features:", X.shape)
print("Target:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])

# All 22 features are categorical
categorical_features = X.columns.tolist()

# Create preprocessing transformer
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)

nb_preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ), categorical_features)
    ]
)

#declaring models
models = {

    "Logistic Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler(with_mean=False)),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ]),

    "Decision Tree": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            random_state=42
        ))
    ]),

    "KNN": Pipeline([
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler(with_mean=False)),
        ("classifier", KNeighborsClassifier(
            n_neighbors=5
        ))
    ]),

    "Naive Bayes": Pipeline([
        ("preprocessor", nb_preprocessor),
        ("classifier", GaussianNB())
    ]),

    "SVM": Pipeline([
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler(with_mean=False)),
        ("classifier", SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ))
    ])
}
trained_models = {}

for name, model in models.items():

    print(f"Training {name}...")

    model.fit(X_train, y_train)

    trained_models[name] = model

    print(f"{name} training completed!\n")
    
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

def evaluate_model(model, X_test, y_test):

    # Generate predictions
    y_pred = model.predict(X_test)

    # Get probability of poisonous class
    classifier = model.named_steps["classifier"]

    positive_index = list(classifier.classes_).index("p")

    y_prob = model.predict_proba(X_test)[:, positive_index]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    auc = roc_auc_score(
        y_test,
        y_prob,
        labels=["e", "p"]
    )

    precision = precision_score(
        y_test,
        y_pred,
        pos_label="p",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label="p",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label="p",
        zero_division=0
    )

    mcc = matthews_corrcoef(y_test, y_pred)

    return {
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    }
    

results = []

for name, model in trained_models.items():

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    metrics["Model"] = name

    results.append(metrics)

    print(f"\n{name}")
    print(metrics)

results_df = pd.DataFrame(results)

results_df = results_df.set_index("Model")

print("\nModel Comparison")
print(results_df.round(4))

plt.figure(figsize=(10, 5))

sns.barplot(
    x=results_df.index,
    y=results_df["Accuracy"]
)

plt.title("Accuracy Comparison of ML Models")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy")

plt.xticks(rotation=25)
plt.tight_layout()
plt.show()

results_df.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Performance Comparison of ML Models")
plt.ylabel("Score")
plt.xlabel("ML Model")

plt.xticks(rotation=25)
plt.legend(loc="lower right")
plt.tight_layout()

plt.show()

for name, model in trained_models.items():

    y_pred = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["e", "p"]
    )

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Edible", "Poisonous"],
        yticklabels=["Edible", "Poisonous"]
    )

    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    plt.show()
    
    for name, model in trained_models.items():

        y_pred = model.predict(X_test)

        print("\n", "=" * 50)
        print(name)
        print("=" * 50)

        print(
            classification_report(
                y_test,
                y_pred,
                target_names=["Edible", "Poisonous"]
            )
        )

best_model = results_df["MCC"].idxmax()

print("Best performing model:", best_model)

print(
    results_df.loc[best_model]
)

import os
import joblib

# Create models directory
os.makedirs("models", exist_ok=True)

# Save all trained pipelines
for name, model in trained_models.items():

    filename = (
        name.lower()
        .replace(" ", "_")
        + ".joblib"
    )

    joblib.dump(
        model,
        f"models/{filename}"
    )

    print(f"Saved: {filename}")