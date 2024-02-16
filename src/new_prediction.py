import numpy as np
from main import * 

# Assuming X is your feature matrix and y is your target variable
# Also assuming decision_tree_model is your DecisionTreeClassifier model

# Feature names
feature_names = X.columns

# Get the input features for a new prediction
new_input_features = []
for feature_name in feature_names:
    value = input(f"{feature_name}: ")
    while value not in ['0', '1']:
        print("Please enter a valid value for feature (0 or 1).")
        value = input(f"{feature_name}: ")
    new_input_features.append(int(value))

new_input_features_array = np.array(new_input_features).reshape(1, -1)
new_input_features_scaled = scaler.transform(new_input_features_array)

# Get the probability for the new input features having endometriosis (class 1)
prediction_proba = decision_tree_model.predict_proba(new_input_features_scaled)
print("The probability of having endometriosis is:", prediction_proba[:, 1])

