from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import pickle

# loading the processed data
processed_df = pd.read_csv("../../data/processed/processed_merge.csv")

# getting the data ready
X = processed_df.drop(columns=["ticket_num"])
y = processed_df["ticket_num"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RF Regressor with hyperparameter tuning
model = RandomForestRegressor(random_state=42)

param_grid = {
    "n_estimators": [50, 100, 200], # number of trees
    "max_depth": [None, 10, 20, 30], # max depth of trees
    "min_samples_split": [2, 5, 10], # min samples for splitting a node
    "min_samples_leaf": [1, 2, 4] # min samples in leaf node
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid,cv=5, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)

print("Best parameters found:", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
#print(y_test.values)
#print(y_pred)
print(f"RF MAE: {mae}, MSE: {mse}, R2: {r2}")
"""
# Polynomial Regression
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(X_train)
model1 = LinearRegression()
model1.fit(x_poly, y_train)
x_test_poly = poly.transform(X_test)
y_pred = model1.predict(x_test_poly)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
#print(y_test.values)
#print(y_pred)
print(f"Poly MAE: {mae}, MSE: {mse}, R2: {r2}")

# save the model to connect to the website later

with open("../../data/processed/model.pkl", "wb") as f:
    pickle.dump(model, f)
"""