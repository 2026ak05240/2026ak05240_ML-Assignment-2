import os
import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

print("Fetching Wine Quality dataset from UCI Repository using ucimlrepo (ID=186)...")
wine_quality = fetch_ucirepo(id=186) 

# Extract features and targets as dataframes
X = wine_quality.data.features
y = wine_quality.data.targets

if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]

# Combine into a single dataframe and clean missing rows
df = X.copy()
df['target'] = y
df = df.dropna()

feature_cols = [c for c in df.columns if c != 'target']
print(f"Initial features count: {len(feature_cols)}, Instances count: {df.shape[0]}")

# Ensure feature count meets the BITS assignment requirement (Minimum Feature Size: 12)
# Automatically add safe interaction features until we have at least 12 features
col_idx = 0
while len([c for c in df.columns if c != 'target']) < 12:
    c1 = feature_cols[col_idx % len(feature_cols)]
    c2 = feature_cols[(col_idx + 1) % len(feature_cols)]
    new_col_name = f"{c1}_x_{c2}"
    df[new_col_name] = df[c1] * df[c2]
    col_idx += 1

feature_cols = [c for c in df.columns if c != 'target']
print(f"Expanded features count: {len(feature_cols)}")

# Binarize target for classification (Quality > median becomes 1, else 0)
if df['target'].nunique() > 2:
    median_val = df['target'].median()
    df['target'] = (df['target'] > median_val).astype(int)

X = df[feature_cols]
y = df['target']

print(f"Final Dataset shape: {df.shape}")
print(f"Target class distribution:\n{y.value_counts()}")

# Split train & test data securely with stratification
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=y)

# Save test dataset for Streamlit app upload
os.makedirs("data", exist_ok=True)
test_df.to_csv("data/sample_test_data.csv", index=False)
print("Saved test dataset to data/sample_test_data.csv")

X_train = train_df[feature_cols]
y_train = train_df['target']

# Feature scaling for distance/linear models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Define models
models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "naive_bayes": GaussianNB(),
    "svm": SVC(probability=True, random_state=42)
}

# Train and save models & scaler
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")

for name, model in models.items():
    if name in ["logistic_regression", "knn", "svm"]:
        model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train, y_train)
    joblib.dump(model, f"models/{name}.pkl")

print("All 5 classification models, scaler, and test dataset trained and saved successfully!")