import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.ensemble import StackingRegressor
import seaborn as sns
import matplotlib.pyplot as plt

# Load the training dataset
data = pd.read_csv("C:\Users\Birendeep SIngh\Documents\cpp\DSA\python\train.csv")

# 1. Check and Handle Missing Values
print("Missing Values:\n", data.isnull().sum())
data = data.dropna()  # Drop rows with missing values

# 2. Remove Outliers using IQR Method
def remove_outliers(df, feature_columns):
    for feature in feature_columns:
        Q1 = df[feature].quantile(0.25)  # First quartile (25%)
        Q3 = df[feature].quantile(0.75)  # Third quartile (75%)
        IQR = Q3 - Q1                    # Interquartile range
        lower_bound = Q1 - 1.5 * IQR     # Lower bound
        upper_bound = Q3 + 1.5 * IQR     # Upper bound
        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    return df

features = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'target']
data = remove_outliers(data, features)

# 3. Check and Drop Duplicates
print(f"Number of duplicate rows before: {data.duplicated().sum()}")
data = data.drop_duplicates()
print(f"Number of duplicate rows after: {data.duplicated().sum()}")

# 4. Feature Scaling
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# 5. Feature Selection using Correlation Matrix
correlation_matrix = data.corr()
print("Correlation Matrix:")
print(correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix Heatmap")
plt.show()

# Drop highly correlated features
data = data.drop(columns=['f2', 'f5'])

# Splitting features and target
X = data.drop(columns=['target'])
y = data['target']

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train Models
# Random Forest
rf_model = RandomForestRegressor(random_state=42, n_estimators=200, max_depth=15)
rf_model.fit(X_train, y_train)
y_val_pred_rf = rf_model.predict(X_val)
r2_rf = r2_score(y_val, y_val_pred_rf)
print(f"Random Forest R² Score: {r2_rf}")

# Gradient Boosting
gbr_model = GradientBoostingRegressor(random_state=42, n_estimators=200, learning_rate=0.05, max_depth=6)
gbr_model.fit(X_train, y_train)
y_val_pred_gbr = gbr_model.predict(X_val)
r2_gbr = r2_score(y_val, y_val_pred_gbr)
print(f"Gradient Boosting R² Score: {r2_gbr}")

# Extra Trees Regressor
etr_model = ExtraTreesRegressor(random_state=42, n_estimators=200, max_depth=15)
etr_model.fit(X_train, y_train)
y_val_pred_etr = etr_model.predict(X_val)
r2_etr = r2_score(y_val, y_val_pred_etr)
print(f"Extra Trees Regressor R² Score: {r2_etr}")

# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
y_val_pred_ridge = ridge_model.predict(X_val)
r2_ridge = r2_score(y_val, y_val_pred_ridge)
print(f"Ridge Regression R² Score: {r2_ridge}")

# 7. Stacking Regressor
stacking_regressor = StackingRegressor(
    estimators=[
        ('rf', rf_model),
        ('gbr', gbr_model),
        ('etr', etr_model),
        ('ridge', ridge_model)
    ],
    final_estimator=GradientBoostingRegressor(random_state=42, n_estimators=100, learning_rate=0.05)
)

stacking_regressor.fit(X_train, y_train)
y_val_pred_stacking = stacking_regressor.predict(X_val)
r2_stacking = r2_score(y_val, y_val_pred_stacking)
print(f"Stacking Regressor R² Score: {r2_stacking}")

# 8. Predict on Test Data
test_data = pd.read_csv("C:\Users\Birendeep SIngh\Documents\cpp\DSA\python\test.csv")
test_ids = test_data['id']
test_data = test_data.drop(columns=['id'])  # Drop 'id'

# Ensure same columns and scaling as training data
test_data = test_data[X.columns]  # Keep only relevant columns
test_data_scaled = scaler.transform(test_data)  # Scale the test data

test_predictions = stacking_regressor.predict(test_data_scaled)

# Create submission file
submission = pd.DataFrame({
    'id': test_ids,
    'target': test_predictions
})
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")
