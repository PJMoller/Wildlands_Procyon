import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import make_scorer
from src.paths import MODELS_DIR, PROCESSED_DIR, PREDICTIONS_DIR, IMG_DIR

os.makedirs(IMG_DIR, exist_ok=True)

def wape_score(y_true, y_pred):
    """Make our own WAPE scorer (negative for minimization)"""
    if np.sum(y_true) == 0:
        return 0.0
    
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)
    return -wape
wape_scorer = make_scorer(wape_score, greater_is_better=True)


def calculate_metrics(y_true, y_pred):
    """Calculate all metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate WAPE (positive value for reporting)
    if np.sum(y_true) == 0:
        wape = np.nan
    else:
        wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100
    
    # Only calculate MAPE for non-zero actuals
    y_true_nonzero = y_true[y_true > 0]
    y_pred_nonzero = y_pred[y_true > 0]
    mape = np.mean(np.abs((y_true_nonzero - y_pred_nonzero) / y_true_nonzero)) * 100 if len(y_true_nonzero) > 0 else np.nan
    
    return {
        'WAPE': wape,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }
    
def create_family_level_models(processed_df, feature_cols):
    """Family-level training with CV on train + held-out 30-day validation"""
    family_models = {}
    family_performance = {}
    
    ticket_families = processed_df['ticket_family'].unique()
    print(f"\nTraining family-level models for {len(ticket_families)} families...")
    
    # Fixed 30-day holdout (true unseen data)
    last_date = processed_df['date'].max()
    validation_start = last_date - pd.Timedelta(days=30)
    
    for family in ticket_families:
        print(f"\nTraining model for family: {family}")
        
        family_data = processed_df[processed_df['ticket_family'] == family].copy()
        family_data = family_data.sort_values('date')
        
        if len(family_data) < 100:
            print(f"Skipping {family} - insufficient data ({len(family_data)} rows)")
            continue
        
        # Split: train (<30 days) + val (last 30 days)
        train_mask = family_data['date'] < validation_start
        val_mask = family_data['date'] >= validation_start
        
        X_train_fam = family_data[train_mask][feature_cols]
        y_train_fam = family_data[train_mask]['ticket_num']
        X_val_fam = family_data[val_mask][feature_cols]
        y_val_fam = family_data[val_mask]['ticket_num']
        
        if len(X_val_fam) < 10:
            print(f"Skipping {family} - not enough validation data")
            continue
        
        # CV ONLY on training data
        X_train_cv = X_train_fam
        y_train_cv = y_train_fam

        n_train_samples = len(X_train_cv)
        n_splits_calc = max(2, n_train_samples // 100)
        n_splits = min(5, n_splits_calc)

        print(f"  {family}: {n_train_samples} rows → {n_splits_calc}→{n_splits} folds")
        print(f"  y_train_cv unique values: {len(y_train_cv.unique())}")
        print(f"  y_train_cv zeros ratio: {(y_train_cv == 0).mean():.1%}")
        # SAFETY CHECK:
        if n_train_samples < 20 * n_splits:
            print(f"  SKIP {family}: too few rows ({n_train_samples} < {20*n_splits} min)")
            continue
        #tscv = TimeSeriesSplit(n_splits = min(5, max(2, len(X_train_cv) // 100)))
        
        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            verbose=-1,
        )
        
        # 1. CV metrics on train data only
        #tscv_train_preds = cross_val_predict(model, X_train_cv, y_train_cv, cv=tscv, method="predict")
        def manual_tscv_predict(model, X, y, n_splits=5):
            preds = np.zeros(len(X))
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            for train_idx, val_idx in tscv.split(X):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
                model.fit(X_tr, y_tr)
                preds[val_idx] = model.predict(X_val)
            return preds

        print(f"  Running manual TS CV...")
        tscv_train_preds = manual_tscv_predict(model, X_train_cv, y_train_cv, n_splits)
        cv_metrics_train = calculate_metrics(y_train_cv, tscv_train_preds)
        
        # 2. Train on train data
        model.fit(X_train_fam, y_train_fam)
        
        # 3. Held-out validation (true unseen)
        y_val_pred = model.predict(X_val_fam)
        val_metrics = calculate_metrics(y_val_fam, y_val_pred)
        
        family_models[family] = model
        family_performance[family] = {
            'cv_train': cv_metrics_train,
            'heldout_val': val_metrics,
            'n_train': len(X_train_fam),
            'n_val': len(X_val_fam)
        }
        
        print(f"Family {family} | CV Train WAPE: {cv_metrics_train['WAPE']:.1f}% | Heldout Val WAPE: {val_metrics['WAPE']:.1f}%")
    
    return family_models, family_performance

def process_data():
    # loading the processed data
    try:
        processed_df = pd.read_csv(PROCESSED_DIR / "processed_merge.csv")
        
        
    except Exception as e:
        print(f"Error loading processed data: {e}")
        return None

    # Define feature columns
    # Exclude target and identifier columns
    exclude_cols = ['ticket_num', 'date', 'ticket_name', 'ticket_family', 'event_name', 'first_sale_date', 'last_sale_date']
    feature_cols = [col for col in processed_df.columns if col not in exclude_cols]
    
    print(f"Total features: {len(feature_cols)}")
    
    
    # Split data ensuring time series order
    processed_df = processed_df.sort_values(['ticket_name', 'date'])
    processed_df['date'] = pd.to_datetime(processed_df['date'], errors='coerce')
    processed_df = processed_df.dropna(subset=['date'])
    
    correlations = processed_df[feature_cols + ['ticket_num']].corr()['ticket_num'].abs().sort_values(ascending=False)
    print(correlations.head(10))

    # Use last 30 days for validation
    last_date = processed_df['date'].max()
    validation_start = pd.to_datetime(last_date) - pd.Timedelta(days=30)
    
    train_df = processed_df[processed_df['date'] < validation_start]
    val_df = processed_df[processed_df['date'] >= validation_start]
    
    X_train = train_df[feature_cols]
    y_train = train_df['ticket_num']
    X_val = val_df[feature_cols]
    y_val = val_df['ticket_num']
    
    # Create sample weights based on ticket family and recency
    sample_weights = np.ones(len(train_df))
    
    # Weight recent data higher
    date_normalized = (train_df['date'] - train_df['date'].min()).dt.days
    recency_weights = 1 + (date_normalized / date_normalized.max()) * 0.5
    sample_weights *= recency_weights
    
    # Weight subscription tickets higher
    subscription_mask = train_df['ticket_family'] == 'subscription'
    sample_weights[subscription_mask] *= 1.2
    
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    
    # --- Train Global Model ---
    print("\nTraining global LightGBM model...")
    
    # Define the model
    model = lgb.LGBMRegressor(
        objective='regression',
        metric='mae',
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1
    )
    
    # Hyperparameter tuning with time series split
#    param_dist = {
#        'num_leaves': [31, 63, 127, 200],
#        'learning_rate': [0.01, 0.05, 0.1],
#        'feature_fraction': [0.7, 0.8, 0.9],
#        'bagging_fraction': [0.7, 0.8, 0.9],
#        'min_child_samples': [20, 50, 100, 200],
#        'reg_alpha': [0, 0.1, 0.5, 1.0],
#        'reg_lambda': [0, 0.1, 0.5, 1.0]
#    }
    
    param_dist = {
        'num_leaves': [31],
        'learning_rate': [0.01],
        'feature_fraction': [0.8],
        'bagging_fraction': [0.7],
        'min_child_samples': [20],
        'reg_alpha': [0.1],
        'reg_lambda': [0]
    }

    tscv = TimeSeriesSplit(n_splits=5)
    random_search = RandomizedSearchCV(
        model, param_distributions=param_dist,
        n_iter=50, cv=tscv,
        scoring=wape_scorer,
        random_state=42, n_jobs=-1,
        verbose=1
    )
    
    # Fit with sample weights
    random_search.fit(X_train, y_train, sample_weight=sample_weights)
    
    best_model = random_search.best_estimator_
    print(f"Best parameters: {random_search.best_params_}")
    
    # --- Evaluate Global Model ---
    print("\nEvaluating global model...")
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    
    train_metrics = calculate_metrics(y_train, y_train_pred)
    val_metrics = calculate_metrics(y_val, y_val_pred)
    
    print("\nGlobal Model Performance:")
    print(f"Train MAE: {train_metrics['MAE']:.2f}, WAPE: {train_metrics['WAPE']:.2f}%")
    print(f"Val MAE: {val_metrics['MAE']:.2f}, WAPE: {val_metrics['WAPE']:.2f}%")
    print(f"Val R²: {val_metrics['R2']:.4f}")
    
    # --- Train Family-Level Models ---
    family_models, family_performance = create_family_level_models(processed_df, feature_cols)
    
    # --- Feature Importance Analysis ---
    print("\nTop 20 Feature Importances (Global Model):")
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(importance_df.head(20).to_string(index=False))
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    importance_df.head(20).plot(kind='barh', x='feature', y='importance', legend=False)
    plt.title('Top 20 Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(IMG_DIR / "feature_importances.png", dpi=300)
    plt.close()
    
    # Save the global model
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(MODELS_DIR / "lgbm_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    
    # Save family models
    with open(MODELS_DIR / "family_models.pkl", "wb") as f:
        pickle.dump(family_models, f)
    
    # Save feature columns
    with open(MODELS_DIR / "feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)
    
    # Save performance metrics
    performance_summary = {
    'global_model': val_metrics,
    'family_models_heldout': {k: v['heldout_val'] for k, v in family_performance.items()},
    'family_models_cv': {k: v['cv_train'] for k, v in family_performance.items()}
    }
    
    with open(MODELS_DIR / "model_performance.pkl", "wb") as f:
        pickle.dump(performance_summary, f)
    
    print("\nModel training completed successfully!")
    print(f"Models saved to ../../data/processed/")
    print(f"Feature importance plot saved to ../../data/processed/feature_importances.png")
    
    # Print family-level performance summary
    if family_performance:
        print("\nFamily-Level Model Performance Summary (HELD-OUT VAL):")
        family_heldout_df = pd.DataFrame(
            {k: v['heldout_val'] for k, v in family_performance.items()}
        ).T
        print(family_heldout_df.round(2).to_string())
        
        # Save BOTH metrics
        family_heldout_df.to_csv(PROCESSED_DIR / "family_heldout_performance.csv")
        pd.DataFrame(family_performance).to_pickle(MODELS_DIR / "family_full_performance.pkl")

if __name__ == "__main__":
    process_data()
