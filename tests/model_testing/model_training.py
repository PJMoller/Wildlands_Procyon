from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import pandas as pd
import pickle

# loading the processed data
processed_df = pd.read_csv("../../data/processed/processed_merge.csv")

# getting the data ready
X = processed_df.drop(columns=["ticket_num"])
y = processed_df["ticket_num"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""
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
# Best parameters found: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
# RF MAE: 62.80674629324547, MSE: 19842.27428233114, R2: 0.8232408697312988
"""

"""
# Polynomial Regression

# A pipeline that first creates polynomial features then applies linear regression
model1_pipe = Pipeline([
    ("poly", PolynomialFeatures()),
    ("ridge", Ridge()),
])

# A parameter grid to search for the best degree of polynomial features
param_grid = {
    "poly__degree": [2],
    "ridge__alpha": [0.01, 0.1, 1.0, 10.0],
}

# Grid search for hyperparameter tuning
grid_search = GridSearchCV(estimator=model1_pipe,param_grid=param_grid,cv=5,
                           scoring="neg_mean_absolute_error", n_jobs=-1, verbose=1)

# Fitting the model
grid_search.fit(X_train, y_train)

print("Best parameters found:", grid_search.best_params_)

# Getting the best model from grid search
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluating the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
#print(y_test.values)
#print(y_pred)
print(f"Poly MAE: {mae}, MSE: {mse}, R2: {r2}")
# Best parameters found: {'poly__degree': 2, 'ridge__alpha': 10.0}
# Poly MAE: 98.619551197128, MSE: 38934.33004683598, R2: 0.6531648429635171
"""
"""
# ElasticNet Regression

model2_elastic = ElasticNet(max_iter=3000)

param_grid = {
    'alpha': [0.01,0.1,0.5,1.0,2.5,5.0,10.0,100.0],    # Overall regularization strength
    'l1_ratio': [0.1,0.3,0.5,0.7,0.9,1.0]   # Balance between the L1 and L2 penalties (Lasso and Ridge)
}

# Grid search for hyperparameter tuning
grid_search = GridSearchCV(estimator=model2_elastic,param_grid=param_grid,cv=5,
                           scoring="neg_mean_absolute_error")

# Fitting the model
grid_search.fit(X_train, y_train)

print("Best parameters found:", grid_search.best_params_)

# Getting the best model from the grid search
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluating the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
#print(y_test.values)
#print(y_pred)
print(f"Elasticnet MAE: {mae}, MSE: {mse}, R2: {r2}")
"""
# SVM Support Vector Machine

svm = SVR()

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

# Grid search for hyperparameter tuning
grid_search = GridSearchCV(estimator=svm,param_grid=param_grid,cv=5,
                           scoring="neg_mean_absolute_error", n_jobs=-1, verbose=3)

# Fitting the model
grid_search.fit(X_train, y_train)

print("Best parameters found:", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluating the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
#print(y_test.values)
#print(y_pred)
print(f"SVM MAE: {mae}, MSE: {mse}, R2: {r2}")
#Best parameters found: {'C': 10, 'gamma': 0.1, 'kernel': 'linear'}
#SVM MAE: 95.35585717162338, MSE: 133774.30084607404, R2: 0.07970436827713079

# save the model to connect to the website later

"""
# XGBoost Regressor with hyperparameter tuning

xgb = XGBRegressor(objective="reg:squarederror", random_state=42)

# Define hyperparameter grid
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 6, 9, 12],
    "learning_rate": [0.01, 0.1, 0.15,0.2],
    "subsample": [0.5, 0.7, 1],
    "colsample_bytree": [0.5,0.7, 1]
}

grid_search = GridSearchCV(xgb, param_grid, scoring="neg_mean_squared_error", cv=5, n_jobs=-1, verbose=3)

grid_search.fit(X_train, y_train)

best_xgb = grid_search.best_estimator_

y_pred = best_xgb.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
#print(y_test.values)
#print(y_pred)
print(grid_search.best_params_)
print(f"XGB MAE: {mae}, MSE: {mse}, R2: {r2}")
# {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 300, 'subsample': 0.7}
# XGB MAE: 63.529842376708984, MSE: 23052.328125, R2: 0.7946450114250183
"""

#with open("../../data/processed/model.pkl", "wb") as f:
#    pickle.dump(model, f)
