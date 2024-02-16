import matplotlib.pyplot as plt
from main import *
from sklearn.linear_model import LogisticRegression

num_features_range = range(1, len(X_train.columns) + 1)

f1_scores = []
auc_scores = []

for num_features in num_features_range:
    selected_features = features_df.head(num_features)['Feature'].tolist()

    # train the model using only selected features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    X_train_selected_scaled = scaler.fit_transform(X_train_selected)
    X_test_selected_scaled = scaler.transform(X_test_selected)

    LogisticRegression_model = LogisticRegression(random_state=42)
    LogisticRegression_model.fit(X_train_selected_scaled, y_train)

    y_pred = LogisticRegression_model.predict(X_test_selected_scaled)

    # we calculate F1 score and AUC
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, LogisticRegression_model.predict_proba(X_test_selected_scaled)[:, 1])
    f1_scores.append(f1)
    auc_scores.append(auc)

plt.figure(figsize=(10, 6))
plt.plot(num_features_range, f1_scores, label='F1 Score')
plt.plot(num_features_range, auc_scores, label='AUC' )
plt.xlabel('Number of Features')
plt.ylabel('Performance')
plt.title('Logistic Regression performance')
plt.legend()
path = os.path.join('../figures', 'logistic_regression_performance.svg')
plt.savefig(path)