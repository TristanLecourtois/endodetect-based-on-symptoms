from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from main import *
from sklearn.metrics import roc_curve, precision_recall_curve, auc



# Define the range of hyperparameters to search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

random_forest_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(random_forest_model, param_grid, cv=10, scoring='f1', n_jobs=-1)

from sklearn.metrics import roc_curve, auc

# Define the range of features to consider
num_features_list = [5, 24, 50]

# Create a subplot for ROC curves
plt.figure(figsize=(10, 6))

for num_features in num_features_list:
    # Select the top num_features important features
    selected_features = features_df.head(num_features)['Feature'].tolist()

    # Train the model using selected features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    # Fit and transform the scaler on the training data
    X_train_selected_scaled = scaler.fit_transform(X_train_selected)
    X_test_selected_scaled = scaler.transform(X_test_selected)

    # Train the Random Forest model
    random_forest_model = RandomForestClassifier(random_state=42)
    random_forest_model.fit(X_train_selected_scaled, y_train)

    # Get predicted probabilities for the positive class
    y_probs = random_forest_model.predict_proba(X_test_selected_scaled)[:, 1]

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve for each set of features
    plt.plot(fpr, tpr, label=f'Num Features = {num_features} (AUC = {roc_auc:.2f})', linewidth=2)

# Plot the diagonal line representing random guessing
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)

# Set plot labels and title
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Recall/Sensitivity)')
# AGenerally, the higher the AUC score, the better a classifier performs for the given task.
plt.title('ROC Curve for Different Numbers of Features')
plt.legend(loc='lower right')
path = os.path.join('../figures', 'ROC_curve.svg')
plt.savefig(path)
plt.show()

