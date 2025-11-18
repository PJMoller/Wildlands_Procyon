from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import lightgbm as lgb
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

def process_data():
    # loading the processed data
    try:
        processed_df = pd.read_csv("../../data/processed/processed_merge.csv")
        holiday_cols = [c for c in processed_df.columns if c.startswith("holiday_")]

        for c in holiday_cols:
            processed_df[c] = (
                processed_df[c]
                .map({True: 1, False: 0, "True": 1, "False": 0})  # handle bool and string
                .fillna(0)
                .astype("int8")
            )
    except Exception as e:
        print(f"Error loading processed data: {e}")
        processed_df = pd.DataFrame()

    if processed_df.empty:
        print("Error: Dataframe is empty")
        return
    else:
        try:    
            processed_df = processed_df.sort_values(by=['year', 'month', 'day']).reset_index(drop=True)

            X = processed_df.drop(columns=["ticket_num"])
            y = processed_df["ticket_num"]
            split_index = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
            y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        except Exception as e:
            print(f"Error during data preparation or split: {e}")
            X_train, X_test, y_train, y_test = None, None, None, None

    return X_train, X_test, y_train, y_test

def analyze_model_errors(model, X_test, y_test, n_largest=20):
    y_pred = model.predict(X_test)

    error_df = X_test.copy()
    

    error_df['actual_sales'] = y_test * np.random.uniform(0.8, 1.2, size=len(y_test))  # Simulate actual sales with some noise
    error_df['predicted_sales'] = np.round(y_pred, 2) # Round predictions for readability
    error_df['absolute_error'] = np.abs(error_df['actual_sales'] - error_df['predicted_sales'])

    sorted_errors = error_df.sort_values(by='absolute_error', ascending=False)

    print(f"Top {n_largest} Prediction Errors:")
    
    display_columns = [
        'actual_sales', 
        'predicted_sales', 
        'absolute_error', 
        'year', 
        'month', 
        'day', 
        'weekday', 
        'temperature',
        'sales_rolling_avg_7',
        'sales_lag_1'
    ]
    
    existing_display_columns = [col for col in display_columns if col in sorted_errors.columns]
    
    print(sorted_errors[existing_display_columns].head(n_largest))
    
    error_analysis_path = "../../data/processed/error_analysis_report.csv"
    sorted_errors.to_csv(error_analysis_path, index=False)
    
    print(f"\n✅ Full error analysis report saved to: {error_analysis_path}")

def randomforest(X_train, X_test, y_train, y_test):
    # RF Regressor with hyperparameter tuning
    try:
        model = RandomForestRegressor(random_state=42)
    
        param_grid = {
            "n_estimators": [50, 100, 200], # number of trees
            "max_depth": [8, 10, 15, 20], # max depth of trees
            "min_samples_split": [2, 5, 10], # min samples for splitting a node
            "max_features": ['sqrt', 0.5, 0.7], # number of features to consider at each split
            "min_samples_leaf": [5, 10, 20] # min samples in leaf node
        }

        # Timeseries cross validation
        tscv = TimeSeriesSplit(n_splits=5)

        grid_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                                         cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=1)

        grid_search.fit(X_train, y_train)

        print("Best parameters found:", grid_search.best_params_)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        y_train_pred = best_model.predict(X_train)

        mae_train = mean_absolute_error(y_train, y_train_pred)
        mse_train = mean_squared_error(y_train, y_train_pred)
        r2_train = r2_score(y_train, y_train_pred)
        #print(y_test.values)
        #print(y_pred)
        print(f"RF MAE: {mae}, MSE: {mse}, R2: {r2}")
        print(f"RF MAE_train: {mae_train}, MSE_train: {mse_train}, R2_train: {r2_train}")

        return best_model
        # Best parameters found: {'n_estimators': 50, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': None}
        # RF MAE: 31.302953009953015, MSE: 9885.767252170544, R2: 0.931991209366365

    except ValueError as e:
        print(f"ValueError: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def polynomial(X_train, X_test, y_train, y_test):
    # Polynomial Regression
    try:
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
        grid_search = RandomizedSearchCV(estimator=model1_pipe,param_distributions=param_grid,cv=5,
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

        return best_model
        # Best parameters found: {'poly__degree': 2, 'ridge__alpha': 10.0}
        # Poly MAE: 98.619551197128, MSE: 38934.33004683598, R2: 0.6531648429635171

    except ValueError as e:
        print(f"ValueError: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None    

def elasticnet(X_train, X_test, y_train, y_test):   
    # ElasticNet Regression
    try:
        model2_elastic = ElasticNet(max_iter=3000)

        param_grid = {
            'alpha': [0.01,0.1,0.5,1.0,2.5,5.0,10.0,100.0],    # Overall regularization strength
            'l1_ratio': [0.1,0.3,0.5,0.7,0.9,1.0]   # Balance between the L1 and L2 penalties (Lasso and Ridge)
        }

        # Grid search for hyperparameter tuning
        grid_search = RandomizedSearchCV(estimator=model2_elastic,param_distributions=param_grid,cv=5,
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

        return best_model
    
    except ValueError as e:
        print(f"ValueError: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None    
    
def svm(X_train, X_test, y_train, y_test):
    # SVM Support Vector Machine
    try:
        svm = SVR()

        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly']
        }

        # Grid search for hyperparameter tuning
        grid_search = RandomizedSearchCV(estimator=svm,param_distributions=param_grid,cv=5,
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

        return best_model
    
        #Best parameters found: {'C': 10, 'gamma': 0.1, 'kernel': 'linear'}
        #SVM MAE: 95.35585717162338, MSE: 133774.30084607404, R2: 0.07970436827713079
    except ValueError as e:
        print(f"ValueError: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None  

# save the model to connect to the website later

def xgboost(X_train, X_test, y_train, y_test):
    # XGBoost Regressor with hyperparameter tuning
    try:
        xgb = XGBRegressor(objective="reg:squarederror", random_state=42)

        # Define hyperparameter grid
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.01, 0.1, 0.15,0.2],
            "subsample": [0.5, 0.7, 1],
            "colsample_bytree": [0.5,0.7, 1],
            "min_child_weight": [1, 5, 10]
        }

        tscv = TimeSeriesSplit(n_splits=5)

        grid_search = RandomizedSearchCV(xgb, param_distributions=param_grid, scoring="neg_mean_squared_error", cv=5, n_jobs=-1, verbose=1, n_iter=25)

        grid_search.fit(X_train, y_train)

        best_xgb = grid_search.best_estimator_

        y_pred = best_xgb.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        #print(y_test.values)
        #print(y_pred)
        print(grid_search.best_params_)
        print(f"XGB test MAE: {mae}, MSE: {mse}, R2: {r2}")

        y_train_pred = best_xgb.predict(X_train)
        mae_train = mean_absolute_error(y_train, y_train_pred)
        mse_train = mean_squared_error(y_train, y_train_pred)
        r2_train = r2_score(y_train, y_train_pred)

        print(f"XGB train test MAE: {mae_train}, MSE: {mse_train}, R2: {r2_train}")

        return best_xgb
        # {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 300, 'subsample': 0.7}
        # XGB MAE: 63.529842376708984, MSE: 23052.328125, R2: 0.7946450114250183
    except ValueError as e:
        print(f"ValueError: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None  


def lightgbm(X_train, X_test, y_train, y_test):
    print("--- Training LightGBM Model ---")
    lgbm = lgb.LGBMRegressor(objective="poisson", random_state=42, early_stopping_round=50)

    param_grid = {
        'n_estimators': [800, 1000],
        'learning_rate': [0.01, 0.03, 0.05],
        'num_leaves': [20, 35, 50],
        'min_child_samples': [50, 100],
        'reg_alpha': [1, 5, 10],
        'reg_lambda': [5, 10, 20],
        'bagging_fraction': [0.6, 0.8, 0.9],
        'bagging_freq': [1, 5],
        'feature_fraction': [0.6, 0.8, 0.9],
    }

    
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = RandomizedSearchCV(estimator=lgbm, param_distributions=param_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1, n_iter=20)
    
    fit_params = {
    "eval_set": [(X_train, y_train)],
    "eval_metric": "mae", # Or "rmse", "l1", "l2"
    "callbacks": [lgb.early_stopping(50, verbose=False)]
    }
    grid_search.fit(X_train, y_train, **fit_params)
    best_LGB = grid_search.best_estimator_

    y_pred = best_LGB.predict(X_test)
    test_with_dates = X_test.copy()
    test_with_dates['actual'] = y_test.values
    test_with_dates['pred'] = y_pred

    # Reconstruct date if you have year, month, day
    test_with_dates['date'] = pd.to_datetime(
        dict(year=test_with_dates['year'],
            month=test_with_dates['month'],
            day=test_with_dates['day'])
    )

    daily = test_with_dates.groupby('date')[['actual', 'pred']].sum()

    wape_daily = (daily['actual'] - daily['pred']).abs().sum() / daily['actual'].abs().sum() * 100
    print(f"Daily total WAPE: {wape_daily:.2f}%")
    wape = np.sum(np.abs(y_test - y_pred)) / np.sum(np.abs(y_test)) * 100
    print(f"LGB test WAPE vs actuals: {wape:.2f}%")


    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(grid_search.best_params_)
    print(f"LGB test MAE: {mae}, MSE: {mse}, R2: {r2}")

    y_train_pred = best_LGB.predict(X_train)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)

    print(f"LGB train test MAE: {mae_train}, MSE: {mse_train}, R2: {r2_train}")

    
    print("LGBM Best Parameters:", grid_search.best_params_)
    analyze_model_errors(best_LGB, X_test, y_test)
    ax = lgb.plot_importance(best_LGB, max_num_features=15, importance_type='gain', title='Feature Importance')

    # Save the plot as PNG
    output_path = "../../data/processed/lgbm_feature_importance.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    return best_LGB


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = process_data()
    best_model = lightgbm(X_train, X_test, y_train, y_test)
    if best_model:
        model_path = "../../data/processed/lgbm_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
            
        print("Model saved successfully.")
    else:
        print("Model training failed. Model not saved.")