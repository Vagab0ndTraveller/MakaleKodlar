import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv('Machine Learning/all_stocks_5yr.csv')

# Select a specific stock (e.g., "AAPL")
stock_data = data[data['Name'] == 'AAPL']

# Create a target column: 1 if close increases, 0 otherwise
stock_data['Target'] = (stock_data['close'].shift(-1) > stock_data['close']).astype(int)

# Drop rows with NaN values (due to shift operation)
stock_data = stock_data.dropna()

# Features and target
X = stock_data[['open', 'high', 'low', 'volume']].values  # Independent variables
y = stock_data['Target'].values  # Dependent variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize classifier
clf = RandomForestClassifier(random_state=42)

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Metrics lists
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
sensitivity_list = []
specificity_list = []
true_positive_rate_list = []
false_positive_rate_list = []

# Cross-validation loop
for train_idx, test_idx in cv.split(X, y):
    X_train_cv, X_test_cv = X[train_idx], X[test_idx]
    y_train_cv, y_test_cv = y[train_idx], y[test_idx]

    # Fit the model
    clf.fit(X_train_cv, y_train_cv)

    # Predictions
    y_pred = clf.predict(X_test_cv)

    # Metrics calculation
    accuracy_list.append(accuracy_score(y_test_cv, y_pred))
    precision_list.append(precision_score(y_test_cv, y_pred, average='binary'))
    recall_list.append(recall_score(y_test_cv, y_pred, average='binary'))
    f1_list.append(f1_score(y_test_cv, y_pred, average='binary'))

    # Confusion matrix for sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(y_test_cv, y_pred).ravel()
    sensitivity_list.append(tp / (tp + fn))  # Sensitivity = Recall
    specificity_list.append(tn / (tn + fp))  # Specificity
    true_positive_rate_list.append(tp / (tp + fn))  # Same as sensitivity
    false_positive_rate_list.append(fp / (fp + tn))

# Print results
print("Accuracy for each fold:", accuracy_list)
print("Mean accuracy:", np.mean(accuracy_list))
print("Precision for each fold:", precision_list)
print("Mean precision:", np.mean(precision_list))
print("Recall for each fold:", recall_list)
print("Mean recall:", np.mean(recall_list))
print("F1 for each fold:", f1_list)
print("Mean F1:", np.mean(f1_list))
print("Sensitivity for each fold:", sensitivity_list)
print("Mean sensitivity:", np.mean(sensitivity_list))
print("Specificity for each fold:", specificity_list)
print("Mean specificity:", np.mean(specificity_list))
print("True positive rate for each fold:", true_positive_rate_list)
print("Mean true positive rate:", np.mean(true_positive_rate_list))
print("False positive rate for each fold:", false_positive_rate_list)
print("Mean false positive rate:", np.mean(false_positive_rate_list))
