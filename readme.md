# Machine Learning Assignment - 2: Classification Model Deployment

## a. Problem Statement
The objective of this assignment is to train, evaluate, and deploy five machine learning classification models on a real-world benchmark dataset from the UCI Machine Learning Repository (Wine Quality Dataset, ID=186) containing at least 12 features and 500 instances. The workflow demonstrates end-to-end ML deployment including model training, evaluation across multiple metrics (Accuracy, AUC, Precision, Recall, F1, and MCC), and an interactive web UI built with Streamlit and deployed via Streamlit Community Cloud.

## b. Dataset Description
- **Dataset Name**: Wine Quality Dataset (UCI Repository ID: 186)
- **Total Instances**: 6,497 samples
- **Feature Size**: Expanded to 13 features (incorporating original physicochemical attributes such as fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, and derived interaction features).
- **Target Variable**: Binary classification label representing wine quality (Binarized around the median score into High/Low quality classes: `0` or `1`).
- **Preprocessing Applied**: Missing value removal, feature expansion for interaction terms, and Standard Scaling for feature normalization on distance-based models (Logistic Regression, KNN, SVM).

## c. GitHub Repository Link
- **GitHub Repository**: `https://github.com/2026ak05240/2026ak05240_ML-Assignment-2`
- **Live Streamlit App**: `https://2026ak05240_ML-Assignment-2.streamlit.app`

## d. Models Used & Evaluation Metrics Comparison

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7462 | 0.8120 | 0.7350 | 0.7612 | 0.7479 | 0.4930 |
| **Decision Tree** | 0.7315 | 0.7890 | 0.7201 | 0.7430 | 0.7314 | 0.4632 |
| **KNN** | 0.7585 | 0.8240 | 0.7489 | 0.7690 | 0.7588 | 0.5174 |
| **Naive Bayes** | 0.7240 | 0.7950 | 0.7102 | 0.7510 | 0.7301 | 0.4489 |
| **Support Vector Machine** | **0.7792** | **0.8490** | **0.7690** | **0.7921** | **0.7804** | **0.5591** |

---

### Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Provided a robust linear baseline with stable convergence on scaled physicochemical features, yielding a solid AUC of $0.8120$. |
| **Decision Tree** | Captured non-linear splits across chemical attributes but experienced minor overfitting on deeper nodes, resulting in a lower MCC ($0.4632$). |
| **KNN** | Performed well by measuring local feature proximity after Standard Scaling, achieving a strong F1 score of $0.7588$. |
| **Naive Bayes** | Executed rapidly with efficient probabilistic estimation, though performance was slightly limited by feature independence assumptions among correlated wine metrics. |
| **Support Vector Machine** | Outperformed all other models across every evaluation metric by constructing optimized non-linear decision boundaries, achieving top Accuracy ($77.92\%$) and MCC ($0.5591$). |
| **Overall Winner for your dataset?** | **Support Vector Machine (SVM)** is the winning model for the Wine Quality dataset due to its superior generalization, highest AUC ($0.8490$), and best overall F1 score. |