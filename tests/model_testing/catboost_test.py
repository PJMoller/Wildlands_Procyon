# pip install openpyxl!!!
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import shap
from sklearn.model_selection import cross_val_score, RepeatedKFold, GridSearchCV, train_test_split, TimeSeriesSplit
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


visitor_og_df = pd.read_csv("../../data/raw/visitordaily.csv", sep=";")
weather_og_df = pd.read_excel("../../data/raw/weather.xlsx")
holiday_og_df = pd.read_excel("../../data/raw/Holidays 2023-2026 Netherlands and Germany.xlsx")

# changes to the visitors df
visitor_og_df.columns = ["groupID","ticket_name", "date", "ticket_num"] # lowercase columns
visitor_og_df["date"] = pd.to_datetime(visitor_og_df["date"], format="%Y-%m-%d") # convert to datetime
visitor_og_df["ticket_num"] = visitor_og_df["ticket_num"].astype(int) # convert to int
visitor_og_df["ticket_name"] = visitor_og_df["ticket_name"].astype(str) # convert to str just to make sure

visitor_og_df = visitor_og_df.drop("groupID", axis=1) # drop access group id since its not needed

# changes to weather df

weather_og_df.columns = ["date", "temperature", "rain", "precipitation", "hour"] # rename columns
weather_og_df = weather_og_df.drop("hour", axis=1)



# changes to the holiday df

# Rename columns so all regions and date are clear
holiday_og_df.columns = ["NLNoord", "NLMidden", "NLZuid", "Niedersachsen", "Nordrhein-Westfalen", "date"]
# Now final_holiday_df can be used for merging, one hot encoded



# merge the 3 datasets, #1 weather + hourly, #2 add holidays

merged_wh_df = pd.merge(weather_og_df, visitor_og_df, on="date", how="inner")

merged_wh_df["date"] = pd.to_datetime(merged_wh_df["date"].dt.date, format="%Y-%m-%d")

# now merge with holidays
merged_final_df = pd.merge(merged_wh_df, holiday_og_df, on="date", how="inner")

merged_final_df["year"] = merged_final_df["date"].dt.year
merged_final_df["month"] = merged_final_df["date"].dt.month
merged_final_df["day"] = merged_final_df["date"].dt.day
merged_final_df["weekday"] = merged_final_df["date"].dt.weekday

scaler = StandardScaler()
num_cols = ["temperature", "rain", "precipitation"]
merged_final_df[num_cols] = scaler.fit_transform(merged_final_df[num_cols])

merged_final_df['temp_rain'] = merged_final_df['temperature'] * merged_final_df['rain']
#merged_final_df['holiday_weekday_interaction'] = (merged_final_df['NLNoord'] == '1').astype(int) * merged_final_df['weekday']

merged_final_df['ticket_num_lag1'] = merged_final_df['ticket_num'].shift(1).bfill()
merged_final_df['ticket_num_lag7'] = merged_final_df['ticket_num'].shift(7).bfill()
merged_final_df['is_weekend'] = merged_final_df['date'].dt.weekday.isin([5, 6]).astype(int)

rare_threshold = 200
value_counts = merged_final_df['ticket_name'].value_counts()
rare_categories = value_counts[value_counts < rare_threshold].index
merged_final_df['ticket_name'] = merged_final_df['ticket_name'].apply(lambda x: 'rare' if x in rare_categories else x)

# final merged df is ready for ML now
#print(merged_final_df.head())

merged_final_df = merged_final_df.sort_values("date")

# Separate features and target
X = merged_final_df.drop(columns=['ticket_num', "date"])  # features
y = merged_final_df['ticket_num']  # target variable

# Identify categorical feature names (all non-numeric columns except target)
categorical_features = ['ticket_name', 'year', 'month', 'day', 'weekday',
                        'NLNoord', 'NLMidden', 'NLZuid', 'Niedersachsen', 'Nordrhein-Westfalen']

X[categorical_features] = X[categorical_features].fillna('missing').astype(str)

# Split data into train and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

split_idx = int(0.8 * len(X))

# Perform temporal split
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Create CatBoost Pool objects with categorical features specified
train_pool = Pool(data=X_train, label=y_train, cat_features=categorical_features)
test_pool = Pool(data=X_test, label=y_test, cat_features=categorical_features)

# Define parameter grid for hyperparameter tuning
param_grid = {
    'iterations': [500, 1000],
    'depth': [6,8],
    'learning_rate': [0.0,1, 0.03, 0.05],
    'l2_leaf_reg': [3, 7, 10],
    'bagging_temperature': [0.5, 1, 2],
    'random_strength': [1, 2],
    "early_stopping_rounds": [50, 100]
}

# Initialize CatBoostRegressor without early stopping (since GridSearchCV uses CV)
model = CatBoostRegressor(cat_features=categorical_features, eval_metric='MAE', random_seed=42, verbose=100)

# Setup GridSearchCV
grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1, verbose=100)

# Fit grid search on raw training data (X_train, y_train)
# Make sure categorical features are still strings or ints – no need for Pool here
grid_search.fit(X_train, y_train)

# Retrieve best model
best_model = grid_search.best_estimator_

# Predict on test set
y_pred = best_model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Best hyperparameters:", grid_search.best_params_)
print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}")
# Best hyperparameters: {'depth': 8, 'iterations': 1500, 'l2_leaf_reg': 3, 'learning_rate': 0.05}
# MAE: 66.6876, MSE: 25235.6306, R2: 0.7752

#Best hyperparameters: {'bagging_temperature': 0.3, 'depth': 10, 'early_stopping_rounds': 50, 'iterations': 1500, 'l2_leaf_reg': 3, 'learning_rate': 0.03, 'random_strength': 2}
#MAE: 72.6952, MSE: 32054.3476, R2: 0.7145

# 1. Feature importance from best model
feature_importances = best_model.get_feature_importance(prettified=True)
print("Feature Importances:\n", feature_importances)

# 2. Repeated k-fold cross-validation for MAE stability check
tss = TimeSeriesSplit(n_splits=5)
cv_mae_scores = -cross_val_score(best_model, X, y, scoring='neg_mean_absolute_error', cv=tss, n_jobs=-1)
print(f"temporal CV MAE Mean: {np.mean(cv_mae_scores):.4f}, Std: {np.std(cv_mae_scores):.4f}")

# 3. Train vs Test error comparison to check overfitting
best_model.fit(X_train, y_train)
y_train_pred = best_model.predict(X_train)
train_mae = mean_absolute_error(y_train, y_train_pred)
print(f"Train MAE: {train_mae:.4f}, Test MAE: {mae:.4f}")

# 4. SHAP values to explain feature effects (for a sample set)
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Print top 5 most important features by mean absolute SHAP value
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_indices = mean_abs_shap.argsort()[-10:][::-1]
top_features = X_test.columns[top_indices]
print("Top 5 features by SHAP importance:", list(top_features))
